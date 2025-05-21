"""
Topology Visualizer for Cloud-Fog-Edge
Questo script visualizza la topologia Cloud-Fog-Edge.
"""

import matplotlib.pyplot as plt
import networkx as nx
import os
import argparse
import logging

# Configurazione del logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedTopology:
    """Versione semplificata della topologia per la visualizzazione."""
    
    def __init__(self, num_fog_nodes=3, num_edge_nodes_per_fog=2):
        self.cloud_nodes = []
        self.fog_nodes = []
        self.edge_nodes = []
        self.registry_node = "registry"
        
        self.num_fog_nodes = num_fog_nodes
        self.num_edge_nodes_per_fog = num_edge_nodes_per_fog
        
        # Crea il grafo
        self.G = nx.Graph()
        
        # Crea la topologia
        self.create_topology()
        
    def create_cloud_node(self):
        """Crea un nodo cloud."""
        cloud_name = "cloud"
        self.cloud_nodes.append(cloud_name)
        self.G.add_node(cloud_name)
        
    def create_fog_nodes(self):
        """Crea nodi fog."""
        for i in range(self.num_fog_nodes):
            fog_name = f"fog-{i}"
            self.fog_nodes.append(fog_name)
            self.G.add_node(fog_name)
            
    def create_edge_nodes(self):
        """Crea nodi edge per ogni fog."""
        for fog_id in range(self.num_fog_nodes):
            for edge_id in range(self.num_edge_nodes_per_fog):
                edge_name = f"edge-{fog_id}-{edge_id}"
                self.edge_nodes.append(edge_name)
                self.G.add_node(edge_name)
                
    def create_registry_node(self):
        """Crea il nodo registry Docker."""
        self.G.add_node(self.registry_node)
        
    def connect_nodes(self):
        """Connette i nodi nella topologia."""
        # Connetti cloud a tutti i fog
        for fog in self.fog_nodes:
            self.G.add_edge("cloud", fog)
        
        # Connetti fog ai propri edge
        for fog_id in range(self.num_fog_nodes):
            fog_name = f"fog-{fog_id}"
            for edge_id in range(self.num_edge_nodes_per_fog):
                edge_name = f"edge-{fog_id}-{edge_id}"
                self.G.add_edge(fog_name, edge_name)
        
        # Connetti fog tra loro
        for i in range(self.num_fog_nodes):
            for j in range(i+1, self.num_fog_nodes):
                self.G.add_edge(f"fog-{i}", f"fog-{j}")
                
    def connect_registry(self, connected_to_all=True):
        """
        Connette il registry ai nodi.
        
        Args:
            connected_to_all: Se True, connette a tutti i nodi. Se False, solo al cloud.
        """
        if connected_to_all:
            # Connetti registry a tutti i nodi
            for node in self.cloud_nodes + self.fog_nodes + self.edge_nodes:
                self.G.add_edge(self.registry_node, node)
        else:
            # Connetti registry solo al cloud
            self.G.add_edge(self.registry_node, "cloud")
        
    def create_topology(self):
        """Crea la topologia completa."""
        self.create_cloud_node()
        self.create_fog_nodes()
        self.create_edge_nodes()
        self.create_registry_node()
        self.connect_nodes()
        
    def visualize(self, title="Cloud-Fog-Edge Topology", with_registry=True, registry_connected_to_all=True, output_dir=None, filename=None):
        """
        Visualizza la topologia.
        
        Args:
            title: Titolo della visualizzazione
            with_registry: Se includere il nodo registry
            registry_connected_to_all: Se il registry è connesso a tutti i nodi o solo al cloud
            output_dir: Directory dove salvare la visualizzazione
            filename: Nome del file
        """
        # Crea una copia del grafo per la visualizzazione
        G_vis = self.G.copy()
        
        if with_registry:
            self.connect_registry(registry_connected_to_all)
        elif self.registry_node in G_vis:
            G_vis.remove_node(self.registry_node)
        
        # Posizioni dei nodi usando un layout spring con seed per riproducibilità
        pos = nx.spring_layout(G_vis, seed=42)
        
        plt.figure(figsize=(12, 10))
        
        # Disegna i nodi con colori diversi
        nx.draw_networkx_nodes(G_vis, pos, nodelist=self.cloud_nodes, 
                              node_color='firebrick', node_size=700, label='Cloud')
        
        nx.draw_networkx_nodes(G_vis, pos, nodelist=self.fog_nodes, 
                              node_color='forestgreen', node_size=500, label='Fog')
        
        nx.draw_networkx_nodes(G_vis, pos, nodelist=self.edge_nodes, 
                              node_color='darkorange', node_size=300, label='Edge')
        
        if with_registry:
            nx.draw_networkx_nodes(G_vis, pos, nodelist=[self.registry_node], 
                                  node_color='royalblue', node_size=600, label='Registry')
        
        # Disegna gli archi
        nx.draw_networkx_edges(G_vis, pos, width=1.0, alpha=0.7)
        
        # Disegna le etichette dei nodi
        nx.draw_networkx_labels(G_vis, pos, font_size=10, font_weight='bold')
        
        plt.title(title)
        plt.axis('off')
        plt.legend(scatterpoints=1)
        plt.tight_layout()
        
        # Salva la visualizzazione se è specificato un nome file
        if filename:
            # Crea la directory di output se non esiste
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filepath = os.path.join(output_dir, filename)
            else:
                filepath = filename
            
            plt.savefig(filepath, dpi=300)
            print(f"Visualizzazione salvata in {filepath}")
        
        return plt

def main():
    # Parse degli argomenti da linea di comando
    parser = argparse.ArgumentParser(description='Genera visualizzazioni della topologia Cloud-Fog-Edge')
    parser.add_argument('--output_dir', type=str, default='topology_images',
                        help='Directory per salvare le visualizzazioni')
    parser.add_argument('--fog_nodes', type=int, default=3,
                        help='Numero di nodi fog (default: 3)')
    parser.add_argument('--edge_per_fog', type=int, default=2,
                        help='Numero di nodi edge per fog (default: 2)')
    parser.add_argument('--show', action='store_true',
                        help='Mostra le visualizzazioni (default: False)')
    
    args = parser.parse_args()
    
    # Crea directory di output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Crea e visualizza topologie con diverse configurazioni
    
    # 1. Topologia standard senza registry
    topology1 = SimplifiedTopology(num_fog_nodes=args.fog_nodes, num_edge_nodes_per_fog=args.edge_per_fog)
    plt1 = topology1.visualize(
        title="Cloud-Fog-Edge Topology\n(without Registry)", 
        with_registry=False,
        output_dir=args.output_dir,
        filename="topology_no_registry.png"
    )
    
    # 2. Topologia con registry connesso solo al cloud
    topology2 = SimplifiedTopology(num_fog_nodes=args.fog_nodes, num_edge_nodes_per_fog=args.edge_per_fog)
    plt2 = topology2.visualize(
        title="Cloud-Fog-Edge Topology\n(Registry connected to Cloud only)", 
        registry_connected_to_all=False,
        output_dir=args.output_dir,
        filename="topology_registry_cloud_only.png"
    )
    
    # 3. Topologia con registry connesso a tutti i nodi
    topology3 = SimplifiedTopology(num_fog_nodes=args.fog_nodes, num_edge_nodes_per_fog=args.edge_per_fog)
    plt3 = topology3.visualize(
        title="Cloud-Fog-Edge Topology\n(Registry connected to all nodes)", 
        registry_connected_to_all=True,
        output_dir=args.output_dir,
        filename="topology_registry_all.png"
    )
    
    # 4. Topologia più grande
    extended_fog = min(args.fog_nodes + 2, 10)  # Aggiungi un paio in più, ma non troppi
    extended_edge = min(args.edge_per_fog + 1, 5)  # Aggiungi un edge in più per fog
    
    topology4 = SimplifiedTopology(num_fog_nodes=extended_fog, num_edge_nodes_per_fog=extended_edge)
    plt4 = topology4.visualize(
        title=f"Extended Cloud-Fog-Edge Topology\n({extended_fog} Fog nodes, {extended_edge} Edge nodes per Fog)", 
        registry_connected_to_all=True,
        output_dir=args.output_dir,
        filename="topology_extended.png"
    )
    
    # Mostra se richiesto
    if args.show:
        plt3.show()
    
    print(f"Tutte le topologie sono state visualizzate e salvate in: {args.output_dir}/")
    print("1. topology_no_registry.png")
    print("2. topology_registry_cloud_only.png")
    print("3. topology_registry_all.png")
    print("4. topology_extended.png")

if __name__ == "__main__":
    main()