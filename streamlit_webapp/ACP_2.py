# Library Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.lines import Line2D
import pandas_profiling as pp
from streamlit_pandas_profiling import st_profile_report

# ML Library Imports
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import cm
from sklearn.metrics import silhouette_samples

# Hide footer and Hamburger Menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Page Header
st.subheader("CHIP690: Foundations of Clinical Data Science")
st.markdown("Applied Concept Paper II")
st.write("*Vibhor Gupta*")
st.write("*2023-03-28*")

#Reading the data file
data_file = st.file_uploader("Please upload the csv data file")

if data_file is None:
	covid_data = pd.read_csv("data_covid.csv")
else:
	covid_data = pd.read_csv(data_file)
	
st.write(covid_data.head())
columns = list(covid_data["measurement_name"].unique())



## Functions to analyze the datasets
def aggregator(df,option="Mean"):
	temp_df = df.iloc[:,[0,5,2,6,7,8]]
	if option == "Mean":
		df_pivot = temp_df.pivot_table(values = "value_as_number", index = ["person_id", "category", "current_age"]
                               , columns = ["measurement_name"])
		df_pivot.reset_index(inplace = True)
	else:
		temp_df.sort_values(by = ["person_id","current_age","category","measurement_name","measurement_date"], inplace=True)
		temp_df.drop_duplicates(subset=["person_id","current_age","category","measurement_name"], keep="first", inplace=True)
		temp_df = temp_df.iloc[:,[0,1,2,3,5]]
		df_pivot = temp_df.pivot_table(values = "value_as_number", index = ["person_id", "category", "current_age"]
                               , columns = ["measurement_name"])
		df_pivot.reset_index(inplace = True)
	return df_pivot

def imputer(df, option = "Simple Imputer"):
	if option == "Simple Imputer":
		imp_mean = SimpleImputer(missing_values=np.nan, strategy = "mean")
		imputed_df = pd.DataFrame(imp_mean.fit_transform(df_pivot.iloc[:,2:]), columns= list(df_pivot.columns[2:]))
	else:
		imp_mean = IterativeImputer(random_state=1234)
		imputed_df = pd.DataFrame(imp_mean.fit_transform(df_pivot.iloc[:,2:]), columns= list(df_pivot.columns[2:]))	
	return(imputed_df)

def clustering_df(df, option = "PCA", feat1='', feat2=''):
	if option == "PCA":
		pca = PCA(random_state=1234, n_components = 2)
		clustering_df = pca.fit_transform(df)
		st.markdown("**Explained variance ratio**")
		st.write("Explained variance ratio for PC1: %s and PC2: %s "%(pca.explained_variance_ratio_[0],pca.explained_variance_ratio_[1]))
	else:
		pass
		clustering_df = df.loc[:,[feat1,feat2]]
	return(clustering_df)

# Creating sidebar for filters
with st.sidebar :
	agg_method = st.selectbox("Aggregation Method",["Mean", "First day data"])
	imputation = st.selectbox("Imputation Method",["Simple Imputer", "MICE"])
	st.write("Clustering Using")
	option = st.selectbox("Method", ["2 Feautures", "PCA"])
	feat1 =''
	feat2 = ''
	if option == "2 Feautures" :
		feat1 = st.selectbox("Feauture 1",columns, index=0)
		feat2 = st.selectbox("Feauture 2",columns, index = 1)

	num_of_clusters = st.slider("Number of Clusters", 1,10,2)

# Finlal Tabs
tab1, tab2, tab3 = st.tabs(["ACP", "Cluster Yourself", "Supplementary Data"])

with tab1 :
	st.markdown("**Introduction**")
	st.markdown("**Materials**")
	st.markdown("**Results**")
	st.markdown("**Discussions**")
	st.markdown("**References**")

with tab2 :
	st.markdown("**Note:** The code works on the default parameters of the standard modules.")
	st.markdown("""<p align = "justify">
		You can view the data analysis, or can play with the filters on the side bar to explore the clustering on your own.
		</p>""", unsafe_allow_html = True)
	df_pivot = aggregator(covid_data, agg_method)
	st.dataframe(df_pivot.head())
	st.caption("Table 1 : Data aggregated by computing %s" %(str.lower(agg_method)))
	st.write("\n")
	imputed_df = imputer(df_pivot, imputation)
	st.dataframe(imputed_df.head())
	st.caption("Table 2 : Data imputation by  %s\n" %(str.lower(imputation)))
	st.write("\n")
	df_cluster = clustering_df(imputed_df, option, feat1, feat2 )
	if option=="2 Feautures":
		st.dataframe(df_cluster.head())
	else:
		st.dataframe(pd.DataFrame(df_cluster).head())
	st.caption("Table 3 : Dataframe for clustering")
	st.write("\n")
	st.markdown("""<p align = "justify">
		Let's find the number of clusters, using the default filter settings and find the optimum K for clustering
		</p>""", unsafe_allow_html = True)

	# Elbow Plot
	distortions = []
	for i in range(1, 11):
	    km = KMeans(n_clusters=i, 
	                init='k-means++', 
	                n_init=10, 
	                max_iter=300, 
	                random_state=0)
	    km.fit(df_cluster)
	    distortions.append(km.inertia_)
	fig1, ax1 = plt.subplots()
	ax1.plot(range(1, 11), distortions, marker='o')
	plt.xlabel('Number of clusters')
	plt.ylabel('Distortion')
	plt.tight_layout()
	st.pyplot(fig1)
	st.caption("Figure 1 : Elbow curve")
	st.write("\n\n")

	# Clustering 
	st.markdown("""<p align = "justify">
		Visualize the clusters
		</p>""", unsafe_allow_html = True)
	km = KMeans(n_clusters= num_of_clusters, 
	            init='k-means++',
	            n_init=10, 
	            max_iter=300,
	            tol=1e-04,
	            random_state=0)
	y_km = km.fit_predict(df_cluster)

	## Visualizing Clusters 
	cluster_labels = np.unique(y_km)
	colormap = matplotlib.cm.get_cmap("Set3").colors[:11]
	markers = list(Line2D.markers.keys())
	fig2,ax2= plt.subplots()
	if option=="2 Feautures":
		X = df_cluster.values
	else:
		X = df_cluster
	for i in cluster_labels:
		ax2.scatter(X[y_km == i, 0],
	            X[y_km == i, 1],
	            s=50, c=colormap[i],
	            marker=markers[i], edgecolor='black',
	            label='Cluster %s'%i)
	ax2.scatter(km.cluster_centers_[:, 0],
	            km.cluster_centers_[:, 1],
	            s=250, marker='*',
	            c='red', edgecolor='black',
	            label='Centroids')
	if option=="2 Feautures":
		plt.xlabel(feat1)
		plt.ylabel(feat2)
	else:
		plt.xlabel("PC1")
		plt.ylabel("PC2")

	plt.legend(scatterpoints=1,loc="best")
	plt.grid()
	plt.tight_layout()
	st.pyplot(fig2)
	st.caption("Figure 2: Clustering")
	st.write("\n\n")

	# silhouette Plot
	st.markdown("""<p align = "justify">
		Check the quality of the clusters using the silhouette plot
		</p>""", unsafe_allow_html = True)
	cluster_labels = np.unique(y_km)
	n_clusters = cluster_labels.shape[0]
	silhouette_vals = silhouette_samples(df_cluster, y_km, metric='euclidean')
	y_ax_lower, y_ax_upper = 0, 0
	yticks = []
	fig3, ax3 = plt.subplots()
	for i, c in enumerate(cluster_labels):
	    c_silhouette_vals = silhouette_vals[y_km == c]
	    c_silhouette_vals.sort()
	    y_ax_upper += len(c_silhouette_vals)
	    color = cm.jet(float(i) / n_clusters)
	    ax3.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
	             edgecolor='none', color=color)

	    yticks.append((y_ax_lower + y_ax_upper) / 2.)
	    y_ax_lower += len(c_silhouette_vals)
	    
	silhouette_avg = np.mean(silhouette_vals)
	plt.axvline(silhouette_avg, color="red", linestyle="--") 
	plt.yticks(yticks, cluster_labels + 1)
	plt.ylabel('Cluster')
	plt.xlabel('Silhouette coefficient')
	plt.tight_layout()
	st.pyplot(fig3)
	st.caption("Figure 3: Silhouette Plot")
	st.write("\n\n")

	# Demographics Table
	data_predicted_labels = pd.merge(df_pivot, pd.DataFrame(y_km, columns=["predicted_label"]), right_index = True, left_index = True )
	demographics = pd.merge(covid_data.drop_duplicates(subset=["person_id"]).iloc[:,[0,1,2,3,4,5]], data_predicted_labels.iloc[:,[0,-1]], left_on = "person_id", right_on = "person_id", how ="inner")
	st.dataframe(pd.crosstab(index=demographics['gen_name'], columns=demographics['predicted_label']))
	st.caption("Table 4 : Gender Based Predicted Cluster Grouping")
	st.write("\n")
	st.dataframe(pd.crosstab(index=demographics['race_name'], columns=demographics['predicted_label']))
	st.caption("Table 5 : Race Based Predicted Cluster Grouping")
	st.write("\n")
	demographics["age_group"] = pd.cut(demographics['current_age'],right=True, bins = [0,17,45,65,200], ordered = True)
	st.dataframe(pd.crosstab(index=demographics['age_group'], columns=demographics['predicted_label']))
	st.caption("Table 6 : Age Based Predicted Cluster Grouping")
	st.write("\n")
	st.dataframe(pd.crosstab(index=demographics['category'], columns=demographics['predicted_label']))
	st.caption("Table 6 : Category Based Predicted Cluster Grouping")
	st.write("\n")


# with tab3:
# 	pr = df_pivot.profile_report()
# 	st_profile_report(pr)







