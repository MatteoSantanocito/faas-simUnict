# Modifiche al file examples/my_simulation/main.py

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

# Importiamo la topologia standard
from sim.topology import Topology
import ether.scenarios.urbansensing as scenario
from ether.core import Node, Capacity, Connection
from sim.docker import DockerRegistry

# Manteniamo gli import del custom scheduler e metrics analyzer
from .custom_scheduler import CustomScheduler
from .metrics_analyzer import MetricsAnalyzer

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import sim.docker
from ether.core import Node
# Salva la funzione originale
original_docker_pull = sim.docker.pull

# Definisci la funzione di patch direttamente qui
def patched_docker_pull(env, image: str, node: Node):
    """
    Versione semplificata di docker pull che simula un trasferimento diretto.
    """
    logger.info(f"Simulando pull di {image} su {node.name}")
    
    # Simula un tempo di trasferimento basato sulla dimensione dell'immagine
    # Prova a ottenere la dimensione dell'immagine dal registry
    size = 58 * 1024 * 1024  # Default: 58MB
    
    try:
        registry = env.container_registry
        for tag_dict in registry.images.values():
            for images in tag_dict.values():
                for img in images:
                    if img.name == image:
                        size = img.size
                        break
    except Exception as e:
        logger.warning(f"Errore nel determinare la dimensione dell'immagine: {e}")
    
    # Simula una velocità di trasferimento di 100 MB/s
    transfer_time = size / (100 * 1024 * 1024)
    logger.info(f"Simulando trasferimento di {size/(1024*1024):.1f} MB in {transfer_time:.2f} secondi")
    
    # Simula il trasferimento
    yield env.timeout(transfer_time)
    logger.info(f"Completato pull di {image} su {node.name}")


class ModifiedTopology(Topology):
    """
    Topologia modificata basata su UrbanSensing con nodi cloud, fog ed edge aggiuntivi.
    """
    def __init__(self, num_fog_nodes=3, num_edge_nodes_per_fog=2, custom_latencies=None):
        super().__init__()
        
        # Configura la topologia base usando lo scenario esistente
        scenario.UrbanSensingScenario().materialize(self)
        
        # Attributi per tenere traccia dei nodi
        self.cloud_nodes = []
        self.fog_nodes = []
        self.edge_nodes = []
        self.custom_latencies = custom_latencies or {}
        
        self.num_fog_nodes = num_fog_nodes
        self.num_edge_nodes_per_fog = num_edge_nodes_per_fog
        
        # Inizializza il registry Docker
        self.init_docker_registry()
        
        # Personalizza la topologia
        self.customize_topology()
        
    def customize_topology(self):
        """
        Estende la topologia con nodi cloud, fog ed edge personalizzati.
        """
        # Aggiungi il nodo cloud
        cloud_node = Node("cloud", Capacity(
            cpu_millis=32000,  # 32 CPU (aumentato)
            memory=64 * 1024 * 1024 * 1024  # 64 GB (aumentato)
        ))
        cloud_node.labels = {"role": "cloud", "zone": "datacenter"}
        self.add_node(cloud_node)
        self.cloud_nodes.append(cloud_node)
        
        # Aggiungi nodi fog
        for i in range(self.num_fog_nodes):
            fog_node = Node(f"fog-{i}", Capacity(
                cpu_millis=16000,  # 16 CPU (aumentato)
                memory=32 * 1024 * 1024 * 1024  # 32 GB (aumentato)
            ))
            fog_node.labels = {"role": "fog", "zone": f"zone-{i}"}
            self.add_node(fog_node)
            self.fog_nodes.append(fog_node)
            
            # Collega cloud a fog
            self.add_edge(cloud_node, fog_node, latency=20, weight=1)
            self.add_edge(fog_node, cloud_node, latency=20, weight=1)
            
            # Aggiungi nodi edge per ogni fog
            for j in range(self.num_edge_nodes_per_fog):
                edge_node = Node(f"edge-{i}-{j}", Capacity(
                    cpu_millis=8000,  # 8 CPU (aumentato)
                    memory=16 * 1024 * 1024 * 1024  # 16 GB (aumentato)
                ))
                edge_node.labels = {
                    "role": "edge", 
                    "zone": f"zone-{i}", 
                    "parent": f"fog-{i}"
                }
                self.add_node(edge_node)
                self.edge_nodes.append(edge_node)
                
                # Collega fog a edge
                self.add_edge(fog_node, edge_node, latency=5, weight=1)
                self.add_edge(edge_node, fog_node, latency=5, weight=1)
        
        # Collega fog tra loro
        for i in range(self.num_fog_nodes):
            for j in range(i + 1, self.num_fog_nodes):
                self.add_edge(self.fog_nodes[i], self.fog_nodes[j], latency=15, weight=1)
                self.add_edge(self.fog_nodes[j], self.fog_nodes[i], latency=15, weight=1)
        
        # Applica latenze personalizzate
        self.apply_custom_latencies()
        
        # Assicurati che tutti i nodi siano collegati al registry
        self.connect_registry_to_all_nodes()
        
        logger.info("Topologia personalizzata creata con successo")
    
    def apply_custom_latencies(self):
        """Applica latenze personalizzate specificate nella configurazione"""
        for key, latency in self.custom_latencies.items():
            try:
                parts = key.split('-')
                if len(parts) >= 2:
                    # Estrai i nomi dei nodi
                    if parts[0] == 'cloud':
                        source = 'cloud'
                        target = '-'.join(parts[1:])
                    else:
                        source = '-'.join(parts[:len(parts)//2])
                        target = '-'.join(parts[len(parts)//2:])
                    
                    # Aggiorna la latenza se esiste un edge
                    edge_data = self.get_edge_data(source, target)
                    if edge_data:
                        edge_data['latency'] = latency
                        logger.debug(f"Aggiornata latenza tra {source} e {target} a {latency}ms")
            except Exception as e:
                logger.warning(f"Errore nell'applicare la latenza personalizzata per {key}: {e}")
    
    def connect_registry_to_all_nodes(self):
        """
        Connette esplicitamente il registry Docker a tutti i nodi della topologia.
        """
        # Trova il nodo registry
        registry = None
        for node in self.nodes():
            if hasattr(node, 'name') and node.name == 'registry':
                registry = node
                break
        
        if not registry:
            logger.warning("Registry Docker non trovato nella topologia!")
            return
        
        # Connetti il registry a tutti i nodi
        connections_added = 0
        for node in self.cloud_nodes + self.fog_nodes + self.edge_nodes:
            if node != registry:
                try:
                    # Usa add_edge per aggiungere l'arco direttamente al grafo NetworkX
                    if not self.has_edge(registry, node):
                        self.add_edge(registry, node, weight=1, latency=5, capacity=1e6)
                        connections_added += 1
                    
                    if not self.has_edge(node, registry):
                        self.add_edge(node, registry, weight=1, latency=5, capacity=1e6)
                        connections_added += 1
                except Exception as e:
                    logger.warning(f"Errore nel collegare registry a {node.name}: {e}")
        
        # Connetti anche ai nodi della topologia base
        for node in self.nodes():
            if node != registry and node not in (self.cloud_nodes + self.fog_nodes + self.edge_nodes):
                try:
                    if not self.has_edge(registry, node):
                        self.add_edge(registry, node, weight=1, latency=5, capacity=1e6)
                        connections_added += 1
                    
                    if not self.has_edge(node, registry):
                        self.add_edge(node, registry, weight=1, latency=5, capacity=1e6)
                        connections_added += 1
                except Exception as e:
                    logger.warning(f"Errore nel collegare registry a {node.name}: {e}")
        
        logger.info(f"Registry Docker connesso a {connections_added//2} nodi in modo bidirezionale")
    
    def get_nodes(self) -> List[Node]:
        """
        Restituisce tutti i nodi nella topologia, escludendo i Link.
        
        Returns:
            Lista di tutti i nodi Node
        """
        # Filtra solo gli oggetti di tipo Node
        return [node for node in self.nodes() 
                if isinstance(node, Node) and hasattr(node, 'capacity')]
    
    def get_nodes_by_role(self, role: str) -> List[Node]:
        """
        Restituisce i nodi per ruolo specifico.
        
        Args:
            role: Ruolo del nodo ('cloud', 'fog', 'edge')
            
        Returns:
            Lista di nodi con il ruolo specificato
        """
        return [node for node in self.get_nodes() if hasattr(node, 'labels') and node.labels.get('role') == role]
    
    def find_node(self, node_name):
        """Trova un nodo per nome."""
        for node in self.nodes():
            if hasattr(node, 'name') and node.name == node_name:
                return node
        return None

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
            request = FunctionRequest(function_name, labels={"source": source_node})
            
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
                "min_replicas": 1,
                "max_replicas": 1
            },
            {
                "name": "resnet50-inference",
                "image": "resnet50-inference-cpu",
                "image_size": 56000000,
                "cpu": 500,
                "memory": 1073741824,
                "weight": 0.5,
                "min_replicas": 1,
                "max_replicas": 1
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
    try:
        # 1. Carica la configurazione
        import argparse
        parser = argparse.ArgumentParser(description='Run a FaaS simulation')
        parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
        args = parser.parse_args()
        config = load_config(args.config)
        
        # 2. Crea la topologia personalizzata
        topology_config = config['topology']
        topology = ModifiedTopology(
            num_fog_nodes=topology_config.get('num_fog_nodes', 3),
            num_edge_nodes_per_fog=topology_config.get('num_edge_nodes_per_fog', 2),
            custom_latencies=topology_config.get('custom_latencies', {})
        )
        
        # 3. Crea il benchmark
        benchmark_config = config['benchmark']
        benchmark = CloudFogEdgeBenchmark(
            function_configs=config['functions'],
            total_requests=benchmark_config.get('total_requests', 500),
            rps=benchmark_config.get('rps', 20),
            request_pattern=benchmark_config.get('request_pattern', 'constant'),
            source_distribution=benchmark_config.get('source_distribution', {'cloud': 1.0})
        )
        
        # 4. Configura la simulazione
        sim = Simulation(topology, benchmark)
        
        # 5. Imposta lo scheduler personalizzato
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
        
        # 6. Esegui la simulazione
        logger.info("Avvio della simulazione...")
        sim.run()
        logger.info("Simulazione completata")
    
        # 7. Analizza i risultati
        metrics = {
            'invocations_df': sim.env.metrics.extract_dataframe('invocations'),
            'allocation_df': sim.env.metrics.extract_dataframe('allocation'),
            'scale_df': sim.env.metrics.extract_dataframe('scale'),
            'node_utilization_df': sim.env.metrics.extract_dataframe('node_utilization'),
            'function_utilization_df': sim.env.metrics.extract_dataframe('function_utilization')
        }
        
        analyzer = MetricsAnalyzer()
        analyzer.analyze_simulation(metrics, config)
        
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione della simulazione: {e}", exc_info=True)
        
if __name__ == "__main__":
    main()