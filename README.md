# Machine-Learning-2-Unsupervised-ML-Project
Credit Card Data Clustering 
# Credit Card Data Clustering

## Overview 
This project aims to cluster credit card data to identify underlying patterns and group similar customer behaviors. By utilizing various clustering algorithms and dimensionality reduction techniques, we can extract insights to improve customer segmentation, detect fraud, and optimize marketing strategies.

## Project Structure

- **Data Loading**: Load the dataset into a suitable data structure.
- **Data Investigation**: Perform exploratory data analysis to understand the structure and distribution of the data.
- **Data Preprocessing**: Clean and preprocess the data to ensure quality and consistency.
- **Feature Transformation**: Apply transformations to the data to prepare for clustering.
- **K-Means Clustering**: Implement K-Means to partition the data into clusters.
- **t-SNE Visualization**: Apply t-SNE to the original data to visualize high-dimensional clusters in a two-dimensional space.
- **Hierarchical Clustering**: Use `scipy.cluster.hierarchy` for hierarchical clustering.
- **PCA vs Kernel PCA**: Compare Principal Component Analysis and Kernel PCA for dimensionality reduction.
- **DBSCAN**: Implement Density-Based Spatial Clustering of Applications with Noise.
- **Visualization of DBSCAN with t-SNE**: Visualize the clusters formed by DBSCAN using t-SNE.
- **EM Algorithm**: Apply the Expectation-Maximization algorithm to estimate cluster parameters and assign data points probabilistically.

## Requirements

This project uses Python 3.8+ and the following libraries:
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib
- seaborn

To install the necessary libraries, you can use the following command:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

## Usage

1. **Data Loading and Preprocessing**:
   - Load the data using Pandas.
   - Clean and preprocess the data (handle missing values, normalize data, etc.).

2. **Feature Transformation**:
   - Apply appropriate transformations, such as scaling or encoding categorical variables.

3. **Clustering Algorithms**:
   - **K-Means**:
     ```python
     from sklearn.cluster import KMeans
     kmeans = KMeans(n_clusters=5)
     kmeans.fit(data)
     ```
   - **Hierarchical Clustering**:
     ```python
     import scipy.cluster.hierarchy as sch
     dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
     ```
   - **DBSCAN**:
     ```python
     from sklearn.cluster import DBSCAN
     dbscan = DBSCAN(eps=0.5, min_samples=5)
     dbscan.fit(data)
     ```

4. **Dimensionality Reduction**:
   - Apply PCA and Kernel PCA, and compare the results.
   - Visualize using t-SNE.

5. **Expectation-Maximization (EM)**:
   - Use Gaussian Mixture Models from `sklearn.mixture` to apply the EM algorithm.

6. **Visualization**:
   - Plot results using `matplotlib` and `seaborn` to visualize the clustering and dimensionality reduction results.


## Conclusion

## Conclusion

The Credit Card Data Clustering project leveraged a variety of clustering techniques and dimensionality reduction methods to analyze and understand customer behaviors from transaction data. Through meticulous data preprocessing and feature transformation, the project ensured a clean and representative dataset was used for analysis. 

The use of multiple clustering algorithms, including K-Means, DBSCAN, and hierarchical clustering, provided a broad perspective on how data points grouped together based on similarities in spending habits and other financial behaviors. The application of both PCA and Kernel PCA allowed us to examine the effectiveness of linear versus non-linear dimensionality reduction in capturing the essence of complex datasets. Moreover, visualizing these clusters through techniques like t-SNE and the use of the EM algorithm offered deeper probabilistic insights into cluster memberships, enhancing the understanding of customer segmentation.

Each clustering method illuminated different aspects of the data. For instance, K-Means provided clear, well-separated groupings assuming spherical clusters, while DBSCAN excelled in identifying outliers and handling clusters of arbitrary shapes. Hierarchical clustering offered an intuitive tree-based representation of data mergers, which was useful for understanding the multi-level similarity structure within the data.

The visualizations, particularly those combining DBSCAN with t-SNE, highlighted the practical challenges and successes of clustering complex real-world data. These insights are invaluable for businesses looking to refine their marketing strategies, improve customer service, or enhance fraud detection systems based on consumer behavior patterns.

Ultimately, this project not only underscores the importance of advanced analytical techniques in strategic decision-making but also sets the stage for further research into more sophisticated algorithms and their applications in the realm of finance and customer data analytics.
