Project Report Title 

Authors Sarah Mueller, Alberto Veloso, Grant Ball  

Abstract: It is a standalone section. It is written to give the reader a summary of your work. Be sure to be specific, yet brief. Even though the abstract comes first in your paper, it is sometimes easier to write the abstract last. (150-300 words) 

	In this project, our team focused on the ever-present problem of poverty in American school districts, particularly that of the state level in the year 2000. The main goal of this project was to analyze which prediction algorithm is better used in different data sources. We will be using this to help determine which schools and given different data and statistics about schools are more susceptible to poverty and difficulties and find the most significant classifiers in determining the probability of a given school facing an underlying poverty issue. To do so we examined the federally available data obtainable via the Common Core of Data (CCD) and performed various regression techniques such as KNN Model, Ridge regression L2-norm, and Lasso regression and found that ______ was the most effective. We then used ____ to determine that (our initial traits) (were/were not) an effective classifier and that ____ was best because of X & Y. 

Introduction 

Your project report is the formal description of your project. The format is similar to the presentation but we want you to fully elaborate on what you did. You could provide some background knowledge about the data you are analyzing, clearly and concisely present your research question, describe the dataset that can help answer your question. (0.5-1 page) 

Brief intro of prompt -> What we hoped to achieve -> Research question/main idea -> Background info/significance -> Our dataset ->  What traits we focused on -> What we used -> ??? 

 

Note: you can transform it to Jupyter Notebook and add sections below as markdown cells. 

Problem Statement 

Give a clear and complete statement of the problem. What is the benchmark you are using. Why? Where does the data come from, what are its characteristics? 

Our main goal with this project is that if given a set of data detailing various niche features about [many] American school(s) we can perform various regression algorithms and analysis to better understand the data and make predictions. The primary focus of these predictions is to evaluate the likelihood of poverty given several stats, remarkably similar to the example given on a recent quiz. The source of our data is the CCD, or as the professor so eloquently put it, “[the] CCD is a comprehensive, annual, national database of all public elementary and secondary schools and school districts”- or in other words, we will be getting our data from publicly available government records. 

Include informal success measures (e.g. accuracy on cross-validated data, without specifying ROC or precision/recall etc) that you planned to use. 

Our initial hypothesis as a group, was to throw the kitchen sink of utilities we have learned from past homework's and lectures at our dataset, and individually analyze each one and determine what we think would be most effective for a given reason, weighing the pros and cons of each. After discussing and reviewing both our notes and the lecture content we came to the initial hypothesis of using KNN regression. We did so primarily because we had the idea reinforced that KNN was an effective route to take when dealing with high dimensional data, and since our data could potentially manipulate dozen(s) of the various given coefficients that seemed like an extremely beneficial trait. We also had agreed as a group that KNN was more effective than some of the more basic functions and would likely be a better fit for our particular data. However, this may not be an ideal solution since one of the drawbacks of KNN is that it gets slower as the size of data grows, and we will need to investigate the data itself and “get our hands dirty before being able to confidentially state that it is the best route forward- in particular those highlighted in homework seven. 

What do you hope to achieve? 

As previously stated, the main goal of our project is to create a set of data (that being correlations and various other metrics to judge the strength of a given coefficient) that will provide new data and help reach new conclusions. More specifically, our team's goal here is to analyze and predict the likelihood of a given school falling under the poverty line when examining a number of features. The features themselves are not entirely evident at this moment, this is because we need to perform various regression algorithms and analysis as we did in the various Homeworks to generate these new variables. With that added information we can come to stronger conclusions and a better understanding of our data, much in the same vein as our work in the past homeworks. After doing all of that, our team will be able to confidently state our newly generated machine-learning guided predictions of poverty with a given probability. 

 

Related Work 

Include background material as appropriate: who cares about this problem, what impact it has, what implications better solutions might have. 

This issue of poverty in America’s youth that our group is tackling in this project is a drastic problem our country faces and has many dire consequences that are quite evident. You do not have to look far to see how underprivileged youths can potentially be abused, or at least miss opportunities that should have been readily available, for instance, basic needs such as food and shelter. People who care about this problem would of course be the children themselves, as well as their parents, guardians, and associated caregivers (be it teachers or simply friends) and to the greater percentage of Americans since the youth grow up to encompass many critical tasks that run our society. Because of this, a better understanding of how poverty works and affects our school systems could have drastic positive impacts on the way our youth grow up, and as a whole raise them to be more effective adults (and potentially alleviate even more problems). 

Included a brief summary of any related work you know about. 

Our group was not immediately certain of any external machine-learning-based poverty statistical analysis, but there were plenty of research papers about the subject. For instance, Stephen Raudenbush’s work with ‘Schooling, Statistics, and Poverty: Can We Measure School Improvement?’ got our group thinking about the underlying implications of a poverty-ridden school (for lack of a better term) and lead us to our comments on the above questions. This work is important because statistics like these are great foundations to base our understanding of the subject. Additionally, they will be crucial when faced with an unknown value resulting from one of our regression functions, we might not understand the correlation between X & Y (for instance ‘FEDERAL REVENUES FROM ALL SOURCES’ & ‘TEACHER SALARIES SPECIAL EDU PROGRAMS’- we might not see the correlation, but our pool of data will!)- but with the qualitative data our benchmarks bring paired with the scientific backing our group will be able to confidentially form a sturdy analysis and conclusion. 

Benchmark implementations - see paperswithcode.com as a good start 

Our group had a tough time initially understanding an appropriate way of intertwining the results/hard data of our various benchmarks alongside the quantitative analysis or cross-references. We looked through various papers on the given website and found excellent examples such as “Multitask Prompted Training Enables Zero-Shot Task Generalization” by Victor Sanh various instances that we could use as a template and better demonstrate our future analysis and conclusions. For instance, on page 8, our group thought that we could generate various graphs similar to that of our past Homeworks and make various observations. Alternatively, papers such as ‘TensorFlow: A system for large-scale machine learning’ by Martín Abadi gave our team the basic idea of not only discussing the qualitative results, but the journey along the way and how we manipulated the various python libraries. 

Data Management 

In this section you should address the questions of interest and interpret the results in terms of the questions of interest you proposed. (1-5 pages, including relevant tables and figures, please adjust your figures to appropriate sizes). 

Describe how did you evaluate your solution 

Our first step will be to analyze the given data and make some preliminary hypotheses on the classifiers and experiment with our KNN benchmark + baseline model to give us a rough approximation of what values are important, and which are dependent on which (like we learned early in the semester). Once we have that data, we can try the other advanced regression techniques learned in the lecture and see which truly performs best and see how we can analyze that qualitative data and see if our hypothesis was confirmed or refuted, and if so, what the real scenario is. 

What evaluation metrics did you use? 

Our idea for what evaluation metrics to use would be to look at accuracy, precision, and recall. Accuracy is one of the most important metrics for every study, we need to check how true the results are from the studied cases. Precision is another especially important metric to check what proportion of the results are truly positive. Lastly, we will use recall that will help find as many positive results as possible. These metrics data will be generated from using the KNN model, ridge regression L2-norm, and lasso regression. We will also use the F1 score to evaluate the results, this is one of the better evaluation metrics because it incorporates both precision and recall by taking the weighted average from both. With the KNN model, we will use different k values to see which value is optimal in the classification of new data. By taking the data already given and using these different models we will be able to decide which is best used.  

Describe a baseline system 

Perhaps use a basic model such as a linear one to get a poor yet somewhat-realistic probability and values to create a baseline model off of. Or more high-level, we’d use a separate model/benchmark to generate values that are far off of desired results, but allow a ‘scale’ to base our ‘real’ findings off of, so that we can better understand them for further manipulation and they aren’t just a high or low number. 

How much did your system outperform the baseline? 

Were there other systems evaluated on the same dataset? How did your system do in comparison to theirs? 

What other regression models will we want to use? KNN, ridge regression L2-norm, and lasso regression 

Show graphs/tables with results 

Error analysis 

Like in HWs 

Suggestions for future improvements 

 

Description of the dataset (dimensions, names of variables with their description) 

Data Gathering 

Answer the questions from Motivation (Sec 3.1.) Composition (Sec 3.2), and Collection (Sec 3.3) of the Datasheets For Datasets paper here. 

If benchmarks, describe the data in details. 

If data collections, justify your methods in terms of data statement. 

What Data Acquisition have you used. Why? (usually algorithms, or data cleaning or wrangling approaches). 

Justify your methods in terms of the problem statement. What did you consider but not use? In particular, be sure to include every method you tried, even if it didn't "work". When describing methods that didn't work, make clear how they failed and any evaluation metrics you used to decide so. How was that a data-driven decision? 

Data Pre-processing, Cleaning, Labeling, and Maintenance 

What Cleaning, and Processing Tools have you used. Why? 

Answer the questions from 3.4 Preprocessing/cleaning/labeling of the Datasheets For Datasets paper here. 

Exploratory Data Analysis 

Describe the methods you explored (usually algorithms, or data wrangling approaches). 

Justify your methods in terms of the problem statement. 

What did you consider but not use? In particular, be sure to include every method you tried, even if it didn't "work". NOTE: Move from .md to .ipynb format when you plan to show EDA (either project proposal or midterm checkpoint) 

Machine Learning Approaches 

In this section, you could describe the methods you used in your analysis. For example, if you are doing classifications, you could introduce the methods like logistic regression, discriminant analysis, support vector machines. You don't have to write formulas if you don't want to do so. It is fine to describe the methods in words. This section basically is a description of the methodologies that you have used for analyzing your data. (up to 2pages) Describe the choice of Machine Learning Tool. Refer ro related work, if applicable. 

Evaluate a primary model and in addition a "baseline" model. 

The baseline is typically the simplest model that's applicable to that data problem 

Naive Bayes for classification 

K-means on raw feature data for clustering. 

Evaluate state-of-art model 

Research GitHub, paperswithcode, Kaggle and similar. 

If not applicable, talk to the instructor. 

Hint Goal is to have some sort of baseline evaluation by Nov 11th checkpoint to establish a scale by which to measure your project's performance. Compare the performance of your baseline model and primary model and explain the differences. 

** This is where all the methods you have tried go, including state-of-art if any ** 

Describe the ML methods that you used and the reasons for their choice. 

What is the family of machine learnign algorithms you are using and why? 

Supervised or Unsupervised? 

Regression or classification? 

Justify ML algorithms in terms of the problem itself and the methods you want to use. 

How did you employ them? 

What features worked well and what didn't? 

Provide documentation for integration 

Tools and Infrastructure Tried and Not Used 

Describe any tools and infrastructure that you tried and ended up not using. What was the problem? Describe infrastructure used. 

Experiments 

Give a detailed summary of the results of your work. 

Setup - Here is where you specify the exact performance measures you used. 

Describe the data used in experiment for presenting dataset: Datasheets for Dataset template 

Describe your accuracy or quality measure, and your performance (runtime or throughput) measure. 

Please use visualizations whenever possible. Include links to interactive visualizations if you built them. 

You can also submit a separated notebook as an appendix to your report if that makes the visualization/interaction task easier. 

It would be reasonable to submit your report as a notebook, but please make sure it runs on one of the two standard environments, and that you include any required files. 

Conclusion 

In this section give a high-level summary of your results. If the reader only reads one section of the report, this one should be it, and it should be self-contained. You can refer back to the Experiments Section for elaborations. This section should be less than a page. In particular emphasize any results that were surprising. 

References 

List the references that cited in your project. 

Appendix## 

Explain the contributions of each member to the project. Include all supporting materials, e.g., additional figures/tables, Python code technical derivations. 
