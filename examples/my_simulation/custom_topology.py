import logging
from typing import Dict, List, Optional

from ether.core import Node, Capacity, Connection
import ether.scenarios.urbansensing as scenario
from sim.topology import Topology

logger = logging.getLogger(__name__)

class CloudFogEdgeTopology(Topology):
    """
    Topologia Cloud-Fog-Edge personalizzata con latenze definite manualmente.
    """
    def __init__(self, num_fog_nodes=3, num_edge_nodes_per_fog=2):
        """
        Inizializza la topologia.
        
        Args:
            num_fog_nodes: Numero di nodi fog
            num_edge_nodes_per_fog: Numero di nodi edge per ogni fog
        """
        super().__init__()
        self.cloud_nodes = []
        self.fog_nodes = []
        self.edge_nodes = []
        
        self.num_fog_nodes = num_fog_nodes
        self.num_edge_nodes_per_fog = num_edge_nodes_per_fog
        
        # Dizionario per tenere traccia delle connessioni e latenze personalizzate
        self.custom_latencies = {}
        
        # Inizializza il registry Docker
        self.init_docker_registry()
        
        # Crea la topologia
        self.create_topology()
        
    def create_topology(self):
        """
        Crea la topologia con nodi cloud, fog ed edge.
        """
        # 1. Crea nodo cloud
        self.create_cloud_node()
        
        # 2. Crea nodi fog
        for i in range(self.num_fog_nodes):
            self.create_fog_node(i)
        
        # 3. Crea nodi edge per ogni fog
        for fog_id in range(self.num_fog_nodes):
            for edge_id in range(self.num_edge_nodes_per_fog):
                self.create_edge_node(fog_id, edge_id)
        
        # 4. Connetti i nodi
        self.connect_nodes()
    
    def create_cloud_node(self):
        """Crea il nodo cloud centrale."""
        cloud = Node("cloud", Capacity(
            cpu_millis=16000,  # 16 CPU
            memory=32 * 1024 * 1024 * 1024  # 32 GB
        ))
        cloud.labels = {"role": "cloud", "zone": "datacenter"}
        self.add_node(cloud)
        self.cloud_nodes.append(cloud)
        return cloud
    
    def create_fog_node(self, fog_id):
        """Crea un nodo fog."""
        fog = Node(f"fog-{fog_id}", Capacity(
            cpu_millis=8000,  # 8 CPU
            memory=16 * 1024 * 1024 * 1024  # 16 GB
        ))
        fog.labels = {"role": "fog", "zone": f"zone-{fog_id}"}
        self.add_node(fog)
        self.fog_nodes.append(fog)
        return fog
    
    def create_edge_node(self, fog_id, edge_id):
        """Crea un nodo edge associato a un fog."""
        edge = Node(f"edge-{fog_id}-{edge_id}", Capacity(
            cpu_millis=4000,  # 4 CPU
            memory=8 * 1024 * 1024 * 1024  # 8 GB
        ))
        edge.labels = {
            "role": "edge", 
            "zone": f"zone-{fog_id}", 
            "parent": f"fog-{fog_id}"
        }
        self.add_node(edge)
        self.edge_nodes.append(edge)
        return edge
    
    def connect_nodes(self):
        """
        Connette i nodi con collegamenti di rete.
        Qui le latenze sono definite in modo statico.
        """
        # Collega cloud a tutti i fog (alto throughput, media latenza)
        for fog in self.fog_nodes:
            self.add_connection(Connection("cloud", fog.name, latency=20))
            # Memorizza la latenza default nel dizionario custom_latencies
            key = tuple(sorted(["cloud", fog.name]))
            self.custom_latencies[key] = 20
        
        # Collega fog ai propri edge (latenza bassa)
        for edge in self.edge_nodes:
            fog_name = edge.labels.get("parent")
            self.add_connection(Connection(fog_name, edge.name, latency=5))
            # Memorizza la latenza default nel dizionario custom_latencies
            key = tuple(sorted([fog_name, edge.name]))
            self.custom_latencies[key] = 5
        
        # Collega fog tra loro (media latenza, medio throughput)
        for i, fog1 in enumerate(self.fog_nodes):
            for j, fog2 in enumerate(self.fog_nodes):
                if i < j:  # Evita collegamenti duplicati
                    self.add_connection(Connection(fog1.name, fog2.name, latency=15))
                    # Memorizza la latenza default nel dizionario custom_latencies
                    key = tuple(sorted([fog1.name, fog2.name]))
                    self.custom_latencies[key] = 15
    
    def set_custom_latency(self, source: str, destination: str, latency: float):
        """
        Imposta una latenza personalizzata tra due nodi.
        Memorizza solo la latenza personalizzata senza tentare di aggiornare le connessioni.
        
        Args:
            source: Nome del nodo sorgente
            destination: Nome del nodo destinazione
            latency: Latenza in millisecondi
        """
        # Crea una chiave ordinata per la mappa delle latenze (source, dest)
        key = tuple(sorted([source, destination]))
        
        # Memorizza la latenza personalizzata (sovrascrive qualsiasi valore esistente)
        self.custom_latencies[key] = latency
    
    def get_latency(self, source: str, destination: str) -> float:
        """
        Ottiene la latenza tra due nodi.
        Restituisce la latenza personalizzata se definita.
        
        Args:
            source: Nome del nodo sorgente
            destination: Nome del nodo destinazione
        
        Returns:
            Latenza in millisecondi
        """
        if source == destination:
            return 0.0
        
        # Controlla se esiste una latenza personalizzata
        key = tuple(sorted([source, destination]))
        if key in self.custom_latencies:
            return self.custom_latencies[key]
        
        # Altrimenti restituisci una latenza di default alta
        return 100.0  # Latenza alta per nodi non direttamente collegati
    
    def get_nodes_by_role(self, role: str) -> List[Node]:
        """
        Restituisce i nodi per ruolo specifico.
        
        Args:
            role: Ruolo del nodo ('cloud', 'fog', 'edge')
            
        Returns:
            Lista di nodi con il ruolo specificato
        """
        return [node for node in self.get_nodes() if node.labels.get('role') == role]
    
    def get_nodes_by_zone(self, zone: str) -> List[Node]:
        """
        Restituisce i nodi per zona specifica.
        
        Args:
            zone: Nome della zona
            
        Returns:
            Lista di nodi nella zona specificata
        """
        return [node for node in self.get_nodes() if node.labels.get('zone') == zone]
    
    def get_nodes(self) -> List[Node]:
        """
        Restituisce tutti i nodi nella topologia.
        
        Returns:
            Lista di tutti i nodi
        """
        return self.cloud_nodes + self.fog_nodes + self.edge_nodes
    
    def get_edge_nodes_for_fog(self, fog_node: Node) -> List[Node]:
        """
        Restituisce i nodi edge associati a un fog.
        
        Args:
            fog_node: Nodo fog
            
        Returns:
            Lista dei nodi edge associati al fog
        """
        return [node for node in self.edge_nodes 
                if node.labels.get('parent') == fog_node.name]