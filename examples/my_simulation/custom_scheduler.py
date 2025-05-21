import logging
import statistics
from typing import List, Optional

from skippy.core.clustercontext import ClusterContext
from skippy.core.model import SchedulingResult, Pod

from sim.core import Environment
from ether.core import Node
from sim.topology import Topology
from sim.docker import DockerRegistry

logger = logging.getLogger(__name__)

class CustomScheduler:
    """
    Scheduler personalizzato con debug avanzato per la simulazione FaaS.
    """
    def __init__(self, cluster: ClusterContext, topology: Topology):
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
        
        # Debug della topologia all'inizializzazione
        self._debug_topology()
        
        # Risorse disponibili per ogni nodo
        self.available_resources = {}
        for node in self.topology.nodes():
            # Verifica che sia un nodo effettivo e non un Link o altro oggetto
            if hasattr(node, 'capacity') and isinstance(node, Node):
                self.available_resources[node.name] = {
                    'cpu_millis': node.capacity.cpu_millis if hasattr(node.capacity, 'cpu_millis') else 8000,
                    'memory': node.capacity.memory if hasattr(node.capacity, 'memory') else 8*1024*1024*1024
                }
        
        logger.debug("Scheduler inizializzato con successo")
    
    def _debug_topology(self):
        """Debug della struttura della topologia"""
        try:
            all_nodes = list(self.topology.nodes())
            topology_nodes = [n for n in all_nodes if isinstance(n, Node)]
            link_nodes = [n for n in all_nodes if not isinstance(n, Node)]
            
            logger.info(f"TOPOLOGIA DEBUG - Totale nodi: {len(all_nodes)}, "
                       f"Nodi effettivi: {len(topology_nodes)}, "
                       f"Link o altro: {len(link_nodes)}")
            
            # Debug dei nodi con nome
            named_nodes = [n for n in topology_nodes if hasattr(n, 'name')]
            
            # Mostra i primi 10 nomi
            if named_nodes:
                node_names = [n.name for n in named_nodes[:10]]
                logger.info(f"Esempi di nodi: {', '.join(node_names)}")
                
            # Verifica se è presente il registry
            registry_node = next((n for n in all_nodes if hasattr(n, 'name') and n.name == 'registry'), None)
            logger.info(f"Registry presente: {'Sì' if registry_node else 'No'}")
            
            # Verifica attributi 'get_nodes' se presenti
            if hasattr(self.topology, 'get_nodes'):
                topology_get_nodes = list(self.topology.get_nodes())
                logger.info(f"Nodi restituiti da get_nodes(): {len(topology_get_nodes)}")
        except Exception as e:
            logger.error(f"Errore nel debug della topologia: {e}")
    
    def schedule(self, pod: Pod) -> SchedulingResult:
        """
        Versione di debug che tenta diversi approcci per trovare un nodo disponibile.
        """
        logger.debug(f"Scheduling pod {pod.name}")
        
        # TENTATIVO 1: Usa i nodi direttamente dalla topologia
        try:
            # Otteniamo i nodi dalla topologia, filtrando solo quelli validi
            topology_nodes = list(self.topology.nodes())
            valid_nodes = [n for n in topology_nodes 
                        if isinstance(n, Node) and 
                           hasattr(n, 'name') and
                           n.name != 'registry' and
                           hasattr(n, 'capacity')]
            
            logger.debug(f"Trovati {len(valid_nodes)} nodi validi nella topologia")
            
            # Se abbiamo nodi validi, usa il primo
            if valid_nodes:
                node = valid_nodes[0]
                logger.info(f"Schedulando {pod.name} su {node.name} (dalla topologia)")
                return SchedulingResult(node, len(valid_nodes), [])
        except Exception as e:
            logger.warning(f"Errore nell'approccio 1 - topologia: {e}")
        
        # TENTATIVO 2: Usa direttamente i nodi cloud/fog/edge se la topologia li ha
        try:
            if hasattr(self.topology, 'cloud_nodes') and self.topology.cloud_nodes:
                node = self.topology.cloud_nodes[0]
                logger.info(f"Schedulando {pod.name} su {node.name} (cloud)")
                return SchedulingResult(node, 1, [])
            
            if hasattr(self.topology, 'fog_nodes') and self.topology.fog_nodes:
                node = self.topology.fog_nodes[0]
                logger.info(f"Schedulando {pod.name} su {node.name} (fog)")
                return SchedulingResult(node, 1, [])
                
            if hasattr(self.topology, 'edge_nodes') and self.topology.edge_nodes:
                node = self.topology.edge_nodes[0]
                logger.info(f"Schedulando {pod.name} su {node.name} (edge)")
                return SchedulingResult(node, 1, [])
        except Exception as e:
            logger.warning(f"Errore nell'approccio 2 - nodi specifici: {e}")
        
        # TENTATIVO 3: Usa i nodi dal cluster
        try:
            cluster_nodes = list(self.cluster.list_nodes())
            logger.debug(f"Trovati {len(cluster_nodes)} nodi nel cluster")
            
            if cluster_nodes:
                for node in cluster_nodes:
                    if hasattr(node, 'name') and node.name != 'registry':
                        logger.info(f"Schedulando {pod.name} su {node.name} (dal cluster)")
                        return SchedulingResult(node, len(cluster_nodes), [])
        except Exception as e:
            logger.warning(f"Errore nell'approccio 3 - cluster: {e}")
        
        # TENTATIVO 4: Crea un nodo artificiale
        try:
            # Questo è un approccio estremo, ma può aiutare a diagnosticare cosa non va
            logger.warning("Creazione di un nodo artificiale per la schedulazione")
            
            from skippy.core.model import Node as SkippyNode
            from skippy.core.model import NodeResources
            
            # Crea un nodo Skippy artificiale
            resources = NodeResources(16, 32 * 1024)  # 16 CPU, 32GB RAM
            artificial_node = SkippyNode("artificial-node", resources)
            
            logger.info(f"Schedulando {pod.name} su {artificial_node.name} (artificiale)")
            return SchedulingResult(artificial_node, 1, ["Nodo artificiale"])
        except Exception as e:
            logger.error(f"Errore nel tentativo di creare un nodo artificiale: {e}")
            
        # Se tutti gli approcci falliscono
        logger.error(f"Tutti gli approcci falliti per {pod.name}")
        
        # Restituisci un risultato vuoto ma valido
        return SchedulingResult(None, 0, ["Nessun nodo trovato dopo tutti i tentativi"])
    
    @staticmethod
    def create(env: Environment, topology: Topology):
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