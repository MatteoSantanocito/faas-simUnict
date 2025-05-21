import logging
import os
import json
from datetime import datetime
from typing import Dict, List
import random

from sim.faassim import Simulation
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import (
    FunctionDeployment, Function, FunctionImage, ScalingConfiguration,
    FunctionContainer, FunctionRequest, KubernetesResourceConfiguration
)
from sim.requestgen import function_trigger, constant_rps_profile, expovariate_arrival_profile, sine_rps_profile

# Importiamo la topologia standard invece di quella personalizzata
from sim.topology import Topology
import ether.scenarios.urbansensing as scenario

# Manteniamo gli import del custom scheduler e metrics analyzer
from .custom_scheduler import CustomScheduler
from .metrics_analyzer import MetricsAnalyzer
from .custom_topology import CloudFogEdgeTopology

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CloudFogEdgeBenchmark(Benchmark):
    """
    Benchmark per simulazioni cloud-fog-edge che utilizza il generatore di richieste di faas-sim.
    """
    def __init__(self, 
                 function_configs: List[Dict],
                 total_requests: int = 500,
                 rps: int = 20,
                 request_pattern: str = 'constant',
                 source_distribution: Dict[str, float] = None):
        """
        Inizializza il benchmark.
        
        Args:
            function_configs: Configurazioni delle funzioni
            total_requests: Numero totale di richieste
            rps: Richieste al secondo
            request_pattern: Pattern di richieste ('constant', 'sine', 'burst')
            source_distribution: Distribuzione delle richieste per sorgente
        """
        super().__init__()
        self.function_configs = function_configs
        self.total_requests = total_requests
        self.rps = rps
        self.request_pattern = request_pattern
        self.source_distribution = source_distribution or {'cloud': 1.0}
        
        # Normalizza la distribuzione delle sorgenti
        total_source_weight = sum(self.source_distribution.values())
        self.source_distribution = {k: v/total_source_weight for k, v in self.source_distribution.items()}
    
    def setup(self, env: Environment):
        """
        Configura l'ambiente con le immagini Docker.
        """
        containers = env.container_registry
        
        # Registra le immagini per ogni funzione configurata
        for fn_config in self.function_configs:
            image_name = fn_config.get('image', f"{fn_config['name']}-cpu")
            image_size = fn_config.get('image_size', 58 * 1024 * 1024)  # Default 58MB
            
            for arch in ["x86", "arm32", "aarch64"]:
                containers.put(ImageProperties(image_name, image_size, arch=arch))
        
        # Log delle immagini registrate
        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info(f"Immagine registrata: {name}, tag: {tag}")
    
    def run(self, env: Environment):
        """
        Esegue il benchmark: deploy delle funzioni e generazione delle richieste.
        """
        # 1. Deploy delle funzioni
        deployments = self.prepare_deployments()
        for deployment in deployments:
            yield from env.faas.deploy(deployment)
        
        # 2. Attendi che le repliche siano disponibili
        for fn_config in self.function_configs:
            fn_name = fn_config['name']
            logger.info(f"Attesa replica disponibile per {fn_name}")
            yield env.process(env.faas.poll_available_replica(fn_name))
        
        # 3. Genera le richieste
        # Calcola la distribuzione delle richieste per funzione
        fn_weights = {fn['name']: fn.get('weight', 1.0) for fn in self.function_configs}
        total_weight = sum(fn_weights.values())
        fn_weights = {k: v/total_weight for k, v in fn_weights.items()}
        
        # Crea un generatore di profilo di carico
        if self.request_pattern == 'sine':
            rps_profile = sine_rps_profile(min_rps=self.rps * 0.5, max_rps=self.rps * 1.5, period=60)
        elif self.request_pattern == 'burst':
            # Alterna tra periodi di carico alto e basso
            rps_values = [self.rps * 0.2] * 20 + [self.rps * 2.0] * 10
            rps_profile = lambda t: rps_values[int(t) % len(rps_values)]
        else:  # 'constant' è il default
            rps_profile = constant_rps_profile(rps=self.rps)
        
        # Genera il generatore di inter-arrivi
        ia_generator = expovariate_arrival_profile(rps_profile)
        
        # Avvia il processo di generazione delle richieste
        count = 0
        while count < self.total_requests:
            # Tempo di inter-arrivo dalla distribuzione
            ia_time = next(ia_generator)
            yield env.timeout(ia_time)
            
            # Seleziona una funzione in base ai pesi
            function_name = random.choices(
                list(fn_weights.keys()),
                weights=list(fn_weights.values())
            )[0]
            
            # Seleziona una sorgente in base alla distribuzione
            source_node = random.choices(
                list(self.source_distribution.keys()),
                weights=list(self.source_distribution.values())
            )[0]
            
            # Crea la richiesta con sorgente come metadato
            request = FunctionRequest(function_name)

            
            # Avvia l'invocazione
            env.process(env.faas.invoke(request))
            count += 1
            
            if count % 50 == 0:
                logger.info(f"Generate {count}/{self.total_requests} richieste")
        
        logger.info(f"Generazione richieste completata: {count} richieste totali")
    
    def prepare_deployments(self) -> List[FunctionDeployment]:
        """
        Prepara i deployment delle funzioni.
        
        Returns:
            Lista di deployment
        """
        deployments = []
        
        for fn_config in self.function_configs:
            fn_name = fn_config['name']
            image_name = fn_config.get('image', f"{fn_name}-cpu")
            
            # Design time: definizione delle immagini e funzioni
            fn_image = FunctionImage(image=image_name)
            function = Function(fn_name, fn_images=[fn_image])
            
            # Run time: configurazione dei container
            cpu = fn_config.get('cpu', 100)  # Default 100m CPU
            memory = fn_config.get('memory', 256 * 1024 * 1024)  # Default 256 MB
            
            # Crea la configurazione delle risorse
            resource_config = KubernetesResourceConfiguration.create_from_str(
                cpu=f"{cpu}m",
                memory=f"{memory // (1024 * 1024)}Mi"
            )
            
            # Crea il container
            fn_container = FunctionContainer(fn_image, resource_config=resource_config)
            
            # Crea il deployment
            min_replicas = fn_config.get('min_replicas', 1)
            max_replicas = fn_config.get('max_replicas', 5)
            
            scaling_config = ScalingConfiguration()
            scaling_config.scale_min = min_replicas
            scaling_config.scale_max = max_replicas
            
            deployment = FunctionDeployment(
                function,
                [fn_container],
                scaling_config
            )
            
            deployments.append(deployment)
        
        return deployments

def load_config(config_file: str = "config.json") -> Dict:
    """
    Carica la configurazione da un file JSON.
    Se il file non esiste, crea una configurazione di default.
    
    Args:
        config_file: Percorso del file di configurazione
        
    Returns:
        Configurazione come dizionario
    """
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Errore nel caricamento della configurazione: {e}")
    
    # Configurazione di default
    default_config = {
        "topology": {
            "num_fog_nodes": 3,
            "num_edge_nodes_per_fog": 2,
            "custom_latencies": {
                "cloud-fog-0": 20,
                "cloud-fog-1": 25,
                "cloud-fog-2": 30,
                "fog-0-edge-0-0": 5,
                "fog-0-edge-0-1": 8,
                "fog-1-edge-1-0": 5,
                "fog-1-edge-1-1": 8,
                "fog-2-edge-2-0": 5,
                "fog-2-edge-2-1": 8,
                "fog-0-fog-1": 15,
                "fog-0-fog-2": 20,
                "fog-1-fog-2": 18
            }
        },
        "functions": [
            {
                "name": "python-pi",
                "image": "python-pi-cpu",
                "image_size": 58000000,
                "cpu": 200,
                "memory": 268435456,
                "weight": 0.5,
                "min_replicas": 2,
                "max_replicas": 5
            },
            {
                "name": "resnet50-inference",
                "image": "resnet50-inference-cpu",
                "image_size": 56000000,
                "cpu": 500,
                "memory": 1073741824,
                "weight": 0.5,
                "min_replicas": 1,
                "max_replicas": 3
            }
        ],
        "benchmark": {
            "total_requests": 100,
            "rps": 10,
            "request_pattern": "constant",
            "source_distribution": {
                "cloud": 0.2,
                "fog-0": 0.3,
                "fog-1": 0.3,
                "fog-2": 0.2
            }
        },
        "scheduler": {
            "weights": {
                "alpha": 0.4,
                "beta": 0.2,
                "gamma": 0.4
            }
        }
    }
    
    # Salva la configurazione di default
    try:
        os.makedirs(os.path.dirname(os.path.abspath(config_file)) or '.', exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        logger.info(f"Configurazione di default salvata in {config_file}")
    except Exception as e:
        logger.error(f"Errore nel salvataggio della configurazione: {e}")
    
    return default_config

def main():
    """
    Esegue la simulazione completa.
    """
    from .docker_patch import patched_pull
    import sim.docker
    original_pull = sim.docker.pull
    sim.docker.pull = patched_pull
    
    try:
        # 1. Carica la configurazione
        config = load_config()
        
        # 2. Crea la topologia personalizzata
        topology_config = config['topology']
        topology = CloudFogEdgeTopology(
            num_fog_nodes=topology_config.get('num_fog_nodes', 3),
            num_edge_nodes_per_fog=topology_config.get('num_edge_nodes_per_fog', 2)
        )
        
        # 3. Imposta le latenze personalizzate
        custom_latencies = topology_config.get('custom_latencies', {})
        for key, latency in custom_latencies.items():
            parts = key.split('-')
            if len(parts) >= 2:
                source, dest = parts[0], '-'.join(parts[1:])
                topology.set_custom_latency(source, dest, latency)
    
        # 4. Crea il benchmark
        benchmark_config = config['benchmark']
        benchmark = CloudFogEdgeBenchmark(
            function_configs=config['functions'],
            total_requests=benchmark_config.get('total_requests', 500),
            rps=benchmark_config.get('rps', 20),
            request_pattern=benchmark_config.get('request_pattern', 'constant'),
            source_distribution=benchmark_config.get('source_distribution', {'cloud': 1.0})
        )
        
        # 5. Configura la simulazione
        sim = Simulation(topology, benchmark)
        
        # 6. Imposta lo scheduler personalizzato
        scheduler_weights = config.get('scheduler', {}).get('weights', {})
        
        def create_scheduler(env):
            scheduler = CustomScheduler.create(env, topology)
            # Imposta i pesi dello scheduler se definiti
            if 'alpha' in scheduler_weights:
                scheduler.alpha = scheduler_weights['alpha']
            if 'beta' in scheduler_weights:
                scheduler.beta = scheduler_weights['beta']
            if 'gamma' in scheduler_weights:
                scheduler.gamma = scheduler_weights['gamma']
            return scheduler
        
        sim.create_scheduler = create_scheduler
        
        # 7. Esegui la simulazione
        logger.info("Avvio della simulazione...")
        sim.run()
        logger.info("Simulazione completata")
    
        # 8. Analizza i risultati
        metrics = {
            'invocations_df': sim.env.metrics.extract_dataframe('invocations'),
            'allocation_df': sim.env.metrics.extract_dataframe('allocation'),
            'scale_df': sim.env.metrics.extract_dataframe('scale'),
            'node_utilization_df': sim.env.metrics.extract_dataframe('node_utilization'),
            'function_utilization_df': sim.env.metrics.extract_dataframe('function_utilization')
        }
        
        analyzer = MetricsAnalyzer()
        analyzer.analyze_simulation(metrics, config)
        
    finally:
        # Ripristina la funzione originale
        sim.docker.pull = original_pull
    
if __name__ == "__main__":
    main()