import logging
import statistics
from typing import List, Optional

from skippy.core.clustercontext import ClusterContext
from skippy.core.model import SchedulingResult, Pod

from sim.core import Environment
from ether.core import Node
from .custom_topology import CloudFogEdgeTopology

logger = logging.getLogger(__name__)

class CustomScheduler:
    """
    Scheduler personalizzato che seleziona il miglior nodo in base a:
    - Utilizzo delle risorse
    - Bilanciamento delle risorse
    - Latenza di rete
    """
    def __init__(self, cluster: ClusterContext, topology: CloudFogEdgeTopology):
        """
        Inizializza lo scheduler.
        
        Args:
            cluster: Contesto del cluster
            topology: Topologia cloud-fog-edge
        """
        self.cluster = cluster
        self.topology = topology
        
        # Pesi per il calcolo del punteggio
        self.alpha = 0.4  # Peso per l'utilizzo delle risorse
        self.beta = 0.2   # Peso per il bilanciamento delle risorse
        self.gamma = 0.4  # Peso per la latenza di rete
        
        # Risorse disponibili per ogni nodo
        self.available_resources = {}
        for node in self.topology.get_nodes():
            self.available_resources[node.name] = {
                'cpu_millis': node.capacity.cpu_millis,
                'memory': node.capacity.memory
            }
        
        logger.debug("Scheduler inizializzato con successo")
    
    def score_nodes(self, nodes: List[Node], source_node: str, pod: Pod) -> Optional[Node]:
        """
        Calcola un punteggio per ogni nodo candidato e seleziona il migliore.
        
        Args:
            nodes: Lista di nodi candidati
            source_node: Nome del nodo sorgente (da cui parte la richiesta)
            pod: Pod da schedulare
            
        Returns:
            Il miglior nodo o None se nessun nodo è adatto
        """
        # Estrai le richieste di risorse dal pod
        cpup = pod.spec.containers[0].resources.requests.get('cpu', 0) / 1000
        memp = pod.spec.containers[0].resources.requests.get('memory', 0) / (1024 * 1024)
        
        resource_scores = {}
        
        for node in nodes:
            name = node.name
            
            # Risorse disponibili
            available = self.available_resources.get(name, {})
            totcpun = available.get('cpu_millis', 0) / 1000
            totmemn = available.get('memory', 0) / (1024 * 1024)
            
            # Calcola risorse residue dopo allocazione
            residual_cpu = totcpun - cpup
            residual_mem = totmemn - memp
            
            # Verifica se ci sono risorse sufficienti
            if residual_cpu < 0 or residual_mem < 0:
                logger.debug(f"Nodo {name} non ha risorse sufficienti per il pod {pod.name}")
                continue
            
            # Calcola la deviazione standard delle risorse residue (bilanciamento)
            residuals = [residual_cpu, residual_mem]
            resource_std = statistics.stdev(residuals) if len(residuals) > 1 else 0
            
            # Normalizza la deviazione standard
            sum_residuals = sum(residuals)
            max_std = 0.5 * sum_residuals if sum_residuals > 0 else 1.0
            normalized_resource_std = min(resource_std / max_std, 1.0)
            
            # Calcola il punteggio di utilizzo
            sum_requests = cpup + memp
            sum_capacity = totcpun + totmemn
            utilization_score = sum_requests / sum_capacity if sum_capacity > 0 else 0
            
            # Calcola il punteggio di latenza
            latency = self.topology.get_latency(source_node, name)
            max_latency = 100.0  # Latenza massima ipotizzata
            normalized_latency = min(latency / max_latency, 1.0)
            latency_score = 1.0 - normalized_latency  # Punteggio più alto per latenza più bassa
            
            # Calcola il punteggio finale
            final_score = (
                self.alpha * utilization_score +
                self.beta * (1 - normalized_resource_std) +
                self.gamma * latency_score
            )
            
            resource_scores[name] = final_score
            
            logger.debug(
                f"Nodo {name} - "
                f"Util={utilization_score:.2f}, "
                f"Bilanciamento={1-normalized_resource_std:.2f}, "
                f"Latenza={latency_score:.2f}, "
                f"Score finale={final_score:.2f}"
            )
        
        if not resource_scores:
            return None
        
        best_node_name = max(resource_scores, key=resource_scores.get)
        return self.topology.find_node(best_node_name)
    
    def update_resources(self, node: Node, pod: Pod):
        """
        Aggiorna le risorse disponibili di un nodo dopo l'allocazione di un pod.
        
        Args:
            node: Nodo su cui è stato allocato il pod
            pod: Pod allocato
        """
        name = node.name
        cpup = pod.spec.containers[0].resources.requests.get('cpu', 0) / 1000
        memp = pod.spec.containers[0].resources.requests.get('memory', 0) / (1024 * 1024)
        
        logger.debug(f"Aggiornamento risorse per {name}: CPU={cpup}, Mem={memp}")
        
        self.available_resources[name]['cpu_millis'] -= cpup * 1000
        self.available_resources[name]['memory'] -= memp * 1024 * 1024
    
    def schedule(self, pod: Pod) -> SchedulingResult:
        """
        Schedula un pod sul miglior nodo disponibile.
        
        Args:
            pod: Pod da schedulare
            
        Returns:
            Risultato della schedulazione
        """
        # Determina il nodo sorgente dalla richiesta (se presente)
        # Nella tua versione di faas-sim, pod non ha l'attributo metadata
        # Usa un valore di default 'cloud'
        source_node = 'cloud'
        
        # Se pod ha labels, prova a estrarre 'source'
        if hasattr(pod, 'labels') and isinstance(pod.labels, dict):
            source_node = pod.labels.get('source', 'cloud')
        
        logger.debug(f"Schedulazione pod {pod.name} con sorgente: {source_node}")
        
        # Ottieni tutti i nodi disponibili
        nodes = self.topology.get_nodes()
        
        # Prova a schedulare sugli edge nodes
        edge_nodes = self.topology.get_nodes_by_role('edge')
        best_edge = self.score_nodes(edge_nodes, source_node, pod)
        
        if best_edge:
            self.update_resources(best_edge, pod)
            logger.info(f"Pod {pod.name} schedulato su edge {best_edge.name}")
            return SchedulingResult(best_edge, len(nodes), [])
        
        # Se non sono disponibili edge, prova con i fog
        fog_nodes = self.topology.get_nodes_by_role('fog')
        best_fog = self.score_nodes(fog_nodes, source_node, pod)
        
        if best_fog:
            self.update_resources(best_fog, pod)
            logger.info(f"Pod {pod.name} schedulato su fog {best_fog.name}")
            return SchedulingResult(best_fog, len(nodes), [])
        
        # Infine, prova con il cloud
        cloud_nodes = self.topology.get_nodes_by_role('cloud')
        best_cloud = self.score_nodes(cloud_nodes, source_node, pod)
        
        if best_cloud:
            self.update_resources(best_cloud, pod)
            logger.info(f"Pod {pod.name} schedulato su cloud {best_cloud.name}")
            return SchedulingResult(best_cloud, len(nodes), [])
        
        # Se non è stato trovato alcun nodo adatto
        logger.warning(f"Nessun nodo disponibile per il pod {pod.name}")
        return SchedulingResult(None, len(nodes), ['Risorse insufficienti'])
    
    @staticmethod
    def create(env: Environment, topology: CloudFogEdgeTopology):
        """
        Factory method per creare un'istanza dello scheduler.
        
        Args:
            env: Ambiente di simulazione
            topology: Topologia cloud-fog-edge
            
        Returns:
            Istanza dello scheduler
        """
        logger.info('Creazione CustomScheduler')
        return CustomScheduler(env.cluster, topology)