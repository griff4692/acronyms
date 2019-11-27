import glob
import re
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class ReductionTechnique(Enum):
    """
    This enumeration lists all the type of techniques that can be used for dimensionality reduction
    """
    PCA = 0


class DimensionalityReducer:
    """
    This class encapsulates all the logic required to reduce the dimensions of the BERT embeddings
    """

    def __init__(self, technique: ReductionTechnique):
        self.pca = PCA(n_components=300)
        self.technique = technique
        self.embeddings = None
        self.reduced_embeddings = None
        self.embeddings_to_file_map = None
        self.file_suffix_pattern = re.compile(r'\.npy$')

    def get_all_embeddings(self, files):
        """
        This method stacks all the embeddings and maintains a map between the stack and the source files
        :param files: The files containing the pickled embeddings
        :return: None
        """
        embeddings = []
        embeddings_to_file_map = list()
        current_embeddings_index = 0
        for file in files:
            file_embeddings = np.load(file)
            embeddings_to_file_map.append((current_embeddings_index, file_embeddings.shape[0], file))
            current_embeddings_index += file_embeddings.shape[0]
            embeddings.append(file_embeddings)
        self.embeddings = np.vstack(embeddings)
        self.embeddings_to_file_map = embeddings_to_file_map

    def perform_pca(self):
        """
        This method performs PCA
        :return: None
        """
        self.reduced_embeddings = self.pca.fit_transform(self.embeddings)

    def reduce_dimensions(self):
        """
        This method performs dimensionality reduction depending on the initialized technique
        :return: None
        """
        if self.technique == ReductionTechnique.PCA:
            self.perform_pca()

    def write_reduced_embeddings(self):
        """
        This method is used to write the reduced embeddings to the disk
        :return: None
        """
        for map_item in self.embeddings_to_file_map:
            start_index, length, file_name = map_item
            end_index = start_index + length
            file_name = self.file_suffix_pattern.sub('_reduced.npy', file_name)
            reduced_file_embedding = self.reduced_embeddings[start_index:end_index]
            np.save(file_name, reduced_file_embedding)

    def optimize_reduced_dimensions_pca(self):
        """
        This method is used to optimize the reduced embeddings dimension
        :return:
        """
        scaler = MinMaxScaler(feature_range=[0, 1])
        embeddings_rescaled = scaler.fit_transform(self.embeddings)
        pca = PCA().fit(embeddings_rescaled)
        # Plotting the Cumulative Summation of the Explained Variance
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')  # for each component
        plt.title('BERT Embedding Dimensions Variance')
        plt.show()


if __name__ == '__main__':
    files = glob.glob('embeddings/lf_embeddings/*.npy')
    files.extend(glob.glob('embeddings/sf_embeddings/*.npy'))
    reducer = DimensionalityReducer(ReductionTechnique.PCA)
    reducer.get_all_embeddings(files)
    reducer.optimize_reduced_dimensions_pca()
    # reducer.reduce_dimensions()
    # reducer.write_reduced_embeddings()
