import numpy as np


class ElasticBand():
    """
    Elastic Band class representing a collection of images between two points.

    Attributes:
    -----------
    initial_point : jnp.array
        The initial point of the elastic band.
    final_point : jnp.array
        The final point of the elastic band.
    special_point : jnp.array
        The special point around which the elastic band is formed.
    n_images : int
        The number of images in the elastic band.
    path : jnp.array
        The computed path containing the images.

    Methods:
    --------
    compute_initial_points(start, end, n_images) -> jnp.array:
        Compute the initial points between start and end.
    """
    def __init__(
            self,
            initial_point: np.ndarray,
            final_point: np.ndarray,
            n_images: int = 50,
            special_point: np.ndarray = None
        ) -> None:
        """
        Initialize the ElasticBand instance.

        Parameters:
        -----------
        initial_point : np.ndarray
            The initial point of the elastic band.
        final_point : np.ndarray
            The final point of the elastic band.
        n_images : int, optional
            The number of images in the elastic band (default is 50).
        special_point : np.ndarray, optional
            The special point around which the elastic band is formed
            (default is None, in which case it is the midpoint between initial_point and final_point).
        """
        self.initial_point = np.array(initial_point)
        self.final_point = np.array(final_point)
        if special_point is None:
            self.special_point = np.array(initial_point) + np.array(final_point)
            self.special_point = self.special_point/2
        else:
            self.special_point = np.array(special_point)
        self.n_images = n_images

        images_1 = self.compute_initial_points(
            self.initial_point, self.special_point, self.n_images//2
        )
        images_2 = self.compute_initial_points(
            self.special_point, self.final_point, self.n_images//2
        )
        self.path = np.vstack([images_1, images_2])

            
    def compute_initial_points(
            self,
            start: np.array,
            end: np.array,
            n_images: int
    ) -> np.array:
        """
        Compute the initial points between start and end.

        Parameters:
        -----------
        start : jnp.array
            The start point.
        end : jnp.array
            The end point.
        n_images : int
            The number of images to compute.

        Returns:
        --------
        jnp.array
            The computed initial points.
        """
        ts = np.linspace(0.0, 1.0, n_images+1)[1:]
        points = [start*(1 - t) + end*t for t in ts]
        return np.stack(points)
