import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(image):
    """
    Extract dominant jersey color from player crop
    """
    pixels = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(pixels)

    return kmeans.cluster_centers_[0]
