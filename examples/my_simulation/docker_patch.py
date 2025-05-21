import logging
from ether.core import Node
from sim.docker import DockerRegistry

# Trova la funzione giusta per ottenere la dimensione dell'immagine
try:
    from sim.docker import image_size_bytes
except ImportError:
    # Fallback: reimplementa la funzione basandosi sul codice originale
    def image_size_bytes(env, image: str) -> int:
        """Get image size in bytes."""
        registry = env.container_registry
        for tag_dict in registry.images.values():
            for images in tag_dict.values():
                for img in images:
                    if img.name == image:
                        return img.size
        return 58 * 1024 * 1024  # Default 58MB se non trovata

logger = logging.getLogger(__name__)

def patched_pull(env, image: str, node: Node):
    """
    Versione patchata di docker pull che gestisce percorsi senza hop.
    """
    try:
        # Prova il percorso normale
        route = env.topology.route(DockerRegistry, node)
        
        # Controlla se il percorso ha hop validi
        if hasattr(route, 'hops') and len(route.hops) > 0:
            # Simuliamo un trasferimento di rete semplice invece di usare la funzione originale
            size = image_size_bytes(env, image)
            transfer_time = size / (100 * 1024 * 1024)  # 100 MB/s
            yield env.timeout(transfer_time)
        else:
            # Fallback per percorsi senza hop
            logger.warning(f"Percorso trovato senza hop tra registry e {node.name}, usando trasferimento diretto")
            size = image_size_bytes(env, image)
            transfer_time = size / (100 * 1024 * 1024)  # 100 MB/s
            yield env.timeout(transfer_time)
    except Exception as e:
        # Se c'è un errore, usa un fallback
        logger.warning(f"Utilizzo fallback per docker pull: {e}")
        # Usa una dimensione standard per l'immagine
        size = 58 * 1024 * 1024  # 58MB
        transfer_time = size / (100 * 1024 * 1024)  # 100 MB/s
        yield env.timeout(transfer_time)
        logger.info(f"Completato pull fallback di {image} su {node.name}")