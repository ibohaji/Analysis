# OBS: This project was made in collaboration with :Piero Rendina (s221440), Alexander Zubov (s223162) 
Data: Feature extraction and visualization

Introduction: 

The data at hand represents a collection of features sampled from a group of Pakistani people that were
hospitalized due to Cardiovascular Heart Disease (CHD) in a late stage (III or IV according to NYHA [8]);
some of them died because of it. The aim of the collection was to assess features associated with heart failure
and ways to predict it, especially with specific circumstances surrounding Pakistani diet [1]. In accordance
with this aim, in [1] Ahmad et al. used different statistical methods to analyze possible links between the
features collected in their survey and the mortality.
The most important features, i.e. the ones with the highest correlation, were age, Ejection Fraction (EF),
Serum Sodium (SNa) and Serum Creatinine (SC). Furthermore, using Kaplan & Meier estimator specific levels
of EF were found to lead to different surviving trends. In summary, the authors assessed the importance of
the features for prediction approaches.
These approaches were revisited by Chiecco & Jurman in [3]. Their goal was to identify an optimal model
to predict mortality based on the gathered observations. To make the approach optimal, they ranked the
features based on their significance. To this end, biostatistical techniques (Mann-Whitney U, Pearson R,
chi square) were employed. They pointed out that EF and SC levels had the highest statistical significance.
Furthermore, different ML algorithms were ranked and exploited to test the predictive capability of those two
variables. The result was that not only they alone sufficed for predicting mortality, but they outperformed
all the others combined. They concluded that these findings might lead to a more effective, less resource
demanding, diagnostics approach for patients with CHD.
Considering the findings above, our goal is to further optimize the prediction of mortality due to heart failure
based on factors associated with CHD. For our regression approach, we choose Serum Creatinine (SC)
as the feature to predict. This is because the previous conclusions suggest that it is the most important
(continuous) variable associated with CHD-based heart failure. Analogous to [3], we will compare different
variables’ significance related to SC, and then choose the most important ones to perform our regression
task. Additionally, like Chicco & Jurman did in [3], we will compare the efficacy of both EF and SC alone,
to check whether only one parameter may classify the patients according to their survival status.
Taken together, these targets will hopefully provide us with a way to further optimize the prediction of heart
failure and a valuable tool for future analyses on bigger and more complex datasets.
The current version of the dataset is provided by the Center for Machine Learning and Intelligent Systems
at the University of California, Irvine (UCI) in their online Machine Learning Repository. This data is used
throughout the report and whole semester project.
