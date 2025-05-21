import logging
from ether.core import Node

logger = logging.getLogger(__name__)

def patched_pull(env, image: str, node: Node):
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