# Project Report Title

Authors Sarah Mueller, Alberto Veloso, Grant Ball  

Abstract: It is a standalone section. It is written to give the reader a summary of your work. Be sure to be specific, yet brief. Even though the abstract comes first in your paper, it is sometimes easier to write the abstract last. (150-300 words) 

In this project, our team focused on the ever-present problem of poverty in American school districts, particularly that of the state level in the year 2000. The main goal of this project was to analyze which prediction algorithm is better used in different data sources. We will be using this to help determine which schools and given different data and statistics about schools are more susceptible to poverty and difficulties and find the most significant classifiers in determining the probability of a given school facing an underlying poverty issue. To do so we examined the federally available data obtainable via the Common Core of Data (CCD) and performed various regression techniques such as KNN Model, Ridge regression L2-norm, and Lasso regression and found that ______ was the most effective. We then used ____ to determine that (our initial traits) (were/were not) an effective classifier and that ____ was best because of X & Y. 

## Introduction

Your project report is the formal description of your project. The format is similar to the presentation but we want you to fully elaborate on what you did. You could provide some background knowledge about the data you are analyzing, clearly and concisely present your research question, describe the dataset that can help answer your question. (0.5-1 page) 

Brief intro of prompt -> What we hoped to achieve -> Research question/main idea -> Background info/significance -> Our dataset ->  What traits we focused on -> What we used -> ??? 

 

Note: you can transform it to Jupyter Notebook and add sections below as markdown cells. 

## Problem Statement 
Our main goal with this project is that if given a set of data detailing various niche features about [many] American school(s) we can perform various regression algorithms and analysis to better understand the data and make predictions. The primary focus of these predictions is to evaluate the likelihood of poverty given several stats, remarkably similar to the example given on a recent quiz. The source of our data is the CCD, or as the professor so eloquently put it, “[the] CCD is a comprehensive, annual, national database of all public elementary and secondary schools and school districts”- or in other words, we will be getting our data from publicly available government records. 

Include informal success measures (e.g. accuracy on cross-validated data, without specifying ROC or precision/recall etc) that you planned to use. 

Our initial hypothesis as a group, was to throw the kitchen sink of utilities we have learned from past homework's and lectures at our dataset, and individually analyze each one and determine what we think would be most effective for a given reason, weighing the pros and cons of each. After discussing and reviewing both our notes and the lecture content we came to the initial hypothesis of using KNN regression. We did so primarily because we had the idea reinforced that KNN was an effective route to take when dealing with high dimensional data, and since our data could potentially manipulate dozen(s) of the various given coefficients that seemed like an extremely beneficial trait. We also had agreed as a group that KNN was more effective than some of the more basic functions and would likely be a better fit for our particular data. However, this may not be an ideal solution since one of the drawbacks of KNN is that it gets slower as the size of data grows, and we will need to investigate the data itself and “get our hands dirty before being able to confidentially state that it is the best route forward- in particular those highlighted in homework seven. 

What do you hope to achieve? 

 As previously stated, the main goal of our project is to create a set of data (that being correlations and various other metrics to judge the strength of a given coefficient) that will provide new data and help reach new conclusions. More specifically, our team's goal here is to analyze and predict the likelihood of a given school falling under the poverty line when examining a number of features. The features themselves are not entirely evident at this moment, this is because we need to perform various regression algorithms and analysis as we did in the various Homeworks to generate these new variables. With that added information we can come to stronger conclusions and a better understanding of our data, much in the same vein as our work in the past homeworks. After doing all of that, our team will be able to confidently state our newly generated machine-learning guided predictions of poverty with a given probability. 

### Related Work

Include background material as appropriate: who cares about this problem, what impact it has, what implications better solutions might have. 

This issue of poverty in America’s youth that our group is tackling in this project is a drastic problem our country faces and has many dire consequences that are quite evident. You do not have to look far to see how underprivileged youths can potentially be abused, or at least miss opportunities that should have been readily available, for instance, basic needs such as food and shelter. People who care about this problem would of course be the children themselves, as well as their parents, guardians, and associated caregivers (be it teachers or simply friends) and to the greater percentage of Americans since the youth grow up to encompass many critical tasks that run our society. Because of this, a better understanding of how poverty works and affects our school systems could have drastic positive impacts on the way our youth grow up, and as a whole raise them to be more effective adults (and potentially alleviate even more problems). 

Included a brief summary of any related work you know about. 

Our group was not immediately certain of any external machine-learning-based poverty statistical analysis, but there were plenty of research papers about the subject. For instance, Stephen Raudenbush’s work with ‘Schooling, Statistics, and Poverty: Can We Measure School Improvement?’ got our group thinking about the underlying implications of a poverty-ridden school (for lack of a better term) and lead us to our comments on the above questions. This work is important because statistics like these are great foundations to base our understanding of the subject. Additionally, they will be crucial when faced with an unknown value resulting from one of our regression functions, we might not understand the correlation between X & Y (for instance ‘FEDERAL REVENUES FROM ALL SOURCES’ & ‘TEACHER SALARIES SPECIAL EDU PROGRAMS’- we might not see the correlation, but our pool of data will!)- but with the qualitative data our benchmarks bring paired with the scientific backing our group will be able to confidentially form a sturdy analysis and conclusion. 

Benchmark implementations - see paperswithcode.com as a good start 

Our group had a tough time initially understanding an appropriate way of intertwining the results/hard data of our various benchmarks alongside the quantitative analysis or cross-references. We looked through various papers on the given website and found excellent examples such as “Multitask Prompted Training Enables Zero-Shot Task Generalization” by Victor Sanh various instances that we could use as a template and better demonstrate our future analysis and conclusions. For instance, on page 8, our group thought that we could generate various graphs similar to that of our past Homeworks and make various observations. Alternatively, papers such as ‘TensorFlow: A system for large-scale machine learning’ by Martín Abadi gave our team the basic idea of not only discussing the qualitative results, but the journey along the way and how we manipulated the various python libraries. 

## Data Management 

In this section you should address the questions of interest and interpret the results in terms of the questions of interest you proposed. (1-5 pages, including relevant tables and figures, please adjust your figures to appropriate sizes). 

Describe how did you evaluate your solution 

Our first step will be to analyze the given data and make some preliminary hypotheses on the classifiers and experiment with our KNN benchmark + baseline model to give us a rough approximation of what values are important, and which are dependent on which (like we learned early in the semester). Once we have that data, we can try the other advanced regression techniques learned in the lecture and see which truly performs best and see how we can analyze that qualitative data and see if our hypothesis was confirmed or refuted, and if so, what the real scenario is. Upon further investigation and analysis though our group decided that a form of logisitic regression would be the best case to start out with, as the outputs of KNN can vary quite drastically- even though logistical convergence is really not at its best when exposed to extremely large datasets (as in # of rows) such as ours. Despite this we were able to get a rough correlation pair-graph for each of our chosen pertinent features, compared against the revenue. Our team decided that obtaining not just a simple revenue but rather a function of revenue divided by students would be the best route; unfortunately the data had a large number of irregularities in the student column, or as the CCD calls it, ‘non-available’ data designated by a ‘-2’. To deal with this issue for the time being we normalized all outliers to 0 even though that could create future issues as our data is trying to overfit to the outliers- but to combat this issue we attempted to manually review our data and choose the pertinent features with the least exposure to such non-availability. 

What evaluation metrics did you use? 

Our idea for what evaluation metrics to use would be to look at accuracy, precision, and recall. Accuracy is one of the most important metrics for every study, we need to check how true the results are from the studied cases. Precision is another especially important metric to check what proportion of the results are truly positive. Lastly, we will use recall that will help find as many positive results as possible. These metrics data will be generated from using the KNN model, ridge regression L2-norm, and lasso regression. We will also use the F1 score to evaluate the results, this is one of the better evaluation metrics because it incorporates both precision and recall by taking the weighted average from both. With the KNN model, we will use different k values to see which value is optimal in the classification of new data. By taking the data already given and using these different models we will be able to decide which is best used.  

Describe a baseline system 

Perhaps use a basic model such as a linear one to get a poor yet somewhat realistic probability and values to create a baseline model off of. Or more high-level, we would use a separate model/benchmark to generate values that are far off from the desired results but allow a ‘scale’ to base our ‘real’ findings off so that we can better understand them for further manipulation, so they are not just a high or low number. 

How much did your system outperform the baseline? 

Were there other systems evaluated on the same dataset? How did your system do in comparison to theirs? 

What other regression models will we want to use? KNN, ridge regression L2-norm 

 

Show graphs/tables with results 

 

 1.


 

This graph demonstrates the best K, it is used in KNN regression to determine the # of nearest neighbors which is the only customizable parameter. It is critical to get an accurate K which is done by obtaining a matrix as demonstrated in the .ipnyb. This number represents the most optimal K for a best-fit model. 

 

2.


 

This confusion matrix shares the same principles of logistical confusion-matrix detailed below, but in this one we can see a more accurate range of values and given that the heatmap is inversed across the board diagonally represents a stronger or better fit. This can be logically explained due to the known fact that our previous logistical model tended to overfit and unperformed, inferring that the KNN model is a better fit for our group’s individual scenario but requires additional analyzation to be certain. 

 

 
3.
 

	 

This confusion matrix demonstrates the effectiveness of our train versus our test data; which can visually represent our model’s tendency to overfit and underperform (due to lack of normalization). 

 4.
 

This pair plot demonstrates the (overfitted) correlation of the various variables in relation to the total state revenue. From this we can say for instance teacher’s salaries are positively related to the revenue; or with more revenue there tends to be higher salaries 

Error analysis 

 At this point in our research, we have not created enough of a prediction model to look at the error analysis, we have only looked at the correlation between the Total State Revenue and our definition of a “poverty” level. However our data is overfitting, so we can predict that there will be a large amount of error.  

Suggestions for future improvements 

In the future, something that could have been changed to improve the quality of our findings was researching what indicates if a school and the surrounding areas are in poverty. While we did some research and used our own experiences to decide what data points we thought were the best indications of poverty. We should have spent more time looking into what statistically is used to determine that. We also where not 100% sure what threshold for the total state revenue line would determine if a school were in poverty, this is another thing we could have done better. The data set also had some errors which made it difficult to decipher what was to be used. Some of the data had the number of students enrolled in the school as negligible, this made it difficult to decide what we could use to decide the poverty line. We decided to look at the total state revenue because we thought it would be the best indicator for this, however we originally did want to look at the revenue per student and use that as a deciding factor, after we divided the total revenue by the total number of students, it was very apparent how this was not a good idea because of the number of zeros recorded in the data. In the future, we should filter through the data a little better to be able to use the revenue per student number.  

Description of the dataset (dimensions, names of variables with their description) 

 Our team decided to utilize the Fiscal, district-level files from 2017-18. This dataset contains a massive amount of information regarding the schools in Texas, but our group decided to focus on financial information which is more valuable in order to achieve our objective. In this dataset is provided the number of students which is the number of students that had their Fall membership confirmed. Total federal, state, and local revenue is the total tax receipts received. The school lunch program shows the amount of revenue each school has available for its lunch programs. Capital outlay and other equipment which is the money available to purchase different equipment for the school. Salaries and employee benefits show the money available for the employee salaries and their benefits. Textbooks costs show the money available for the purchase of textbooks. In order to predict the likelihood of a school going through poverty, our team decided to analyze the Total State Revenue and create a certain amount that would help us decide if the school is in an advantageous position financially or not. We created a column “Poverty” in which we would compare the value of Total State Revenue with our amount created to determine if the school is financially stable which is $8.216.000. If a school has a Total State revenue below this amount it will get a “1” and the ones above it will get a “2”. 1 is signalizing the school is falling under what would be designated as “Poverty” and “2” signalizes the schools that are financially stable.
### Data Gathering

Answer the questions from Motivation (Sec 3.1.) Composition (Sec 3.2), and Collection (Sec 3.3) of the Datasheets For Datasets paper here. 

If benchmarks, describe the data in details. 

If data collections, justify your methods in terms of data statement. 

What Data Acquisition have you used. Why? (usually algorithms, or data cleaning or wrangling approaches). 

Justify your methods in terms of the problem statement. What did you consider but not use? In particular, be sure to include every method you tried, even if it didn't "work". When describing methods that didn't work, make clear how they failed and any evaluation metrics you used to decide so. How was that a data-driven decision? 

### Data Pre-processing, Cleaning, Labeling, and Maintenance 

What Cleaning, and Processing Tools have you used.  Why? 
* Answer the questions from 3.4 **Preprocessing/cleaning/labeling** of the [Datasheets For Datasets](https://arxiv.org/abs/1803.09010) paper here. 

### Exploratory Data Analysis 

Describe the methods you explored (usually algorithms, or data wrangling approaches). 

	Our teams first idea was to utilize KNN classification but upon further delving into the data, paired with our newly obtained knowledge from the midterm we quickly came to realize that the KNN method of machine learning was unsuitable for our data, particularly because of the immense dimensionality of the data. KNN does not work well when using data with high dimensionality because it needs all points to be close to each other on every axis of the data set. Because of this we decided to focus our data and separate the data set into a true/false column, once the total state revenue dipped below a certain number we would deem the school as being below the poverty line. By doing this we were able to decide and predict in the future which schools would be below the poverty line.  

Justify your methods in terms of the problem statement. 

What did you consider but not use? In particular, be sure to include every method you tried, even if it didn't "work". NOTE: Move from .md to .ipynb format when you plan to show EDA (either project proposal or midterm checkpoint) 

## Machine Learning Approaches

In this section, you could describe the methods you used in your analysis. For example, if you are doing classifications, you could introduce the methods like logistic regression, discriminant analysis, support vector machines. You don't have to write formulas if you don't want to do so. It is fine to describe the methods in words. This section basically is a description of the methodologies that you have used for analyzing your data. (up to 2pages) Describe the choice of Machine Learning Tool. Refer to related work, if applicable. 

The broadest topic in terms of scope in this report was the decision of regression versus classification; this is a critical decision in machine learning as it is the first step in deciding what algorithm or machine learning model you want to utilize. Our group decided that classification would be the best route to go because such approaches excel at outputting a value between 0 and 1 to give you a % value of your prediction. This would allow us to examine a school’s traits and determine a probability of their finances, which fits perfectly with the main point of this entire point of this report. 

The next step would be deciding what form of classification to use. Our group decided that either KNN or logistical would be the most viable approach as a baseline model after a long argument over the pros and cons of what would be best when and why. The summary of our conclusion (detailed in depth below) is that KNN is better for non-linear datasets but lacks customization and principal component analysis, whereas logistical has a number of additional problems such as underperforming on large datasets but is able to, on average, out perform linear classification on non-linear datasets which is what led us to our decision. 

Evaluate a primary model and in addition a "baseline" model. 

The baseline is typically the simplest model that's applicable to that data problem 

Naive Bayes for classification 

K-means on raw feature data for clustering. 

Evaluate state-of-art model 

Research GitHub, paperswithcode, Kaggle and similar. 

If not applicable, talk to the instructor. 

Hint Goal is to have some sort of baseline evaluation by Nov 11th checkpoint to establish a scale by which to measure your project's performance. Compare the performance of your baseline model and primary model and explain the differences. 

** This is where all the methods you have tried go, including state-of-art if any ** 


### Describe the ML methods that you used and the reasons for their choice. 
What is the family of machine learning algorithms you are using and why? 

Supervised or Unsupervised? 

Our group decided to focus on the Supervised aspect of machine learning, primarily because we have not covered unsupervised in lecture and we thought that we could best utilize the libraries and functions provided in more familiar algorithms such as (but not limited to): KNN, LDA, Logistic, linear, and random Forrest trees. While unsupervised approaches may give many benefits, they are frankly ones we are unaware of and would not be able to properly utilize. 

Regression or classification? 

After discussing it as a group we decided that a classification-based approach would be the most useful in this scenario. This is because our group’s approach is to grab a few pertinent columns, and utilize various forms of machine learning algorithms on them to analyze the results. More specifically, we want to examine the potential existence of correlations/relations between the different variables, such as `Teacher’s Salary` & `Federal Grants available` and see if we can’t make a prediction on whether or not this school will fall under what would be considered the poverty line; or in other words to give a classification based result, a number between 0 and 1 that indicates a probability that a given school falls under the poverty line given simply a few seemingly random, yet correlated, variables.  

### Justify ML algorithms in terms of the problem itself and the methods you want to use. 
How did you employ them? 

What features worked well and what didn't? 

One feature our group decided to [attempt to] employ was Normalization, or the concept of reducing various #s to a relatable range to prevent outliers from skewing your data and causing your model to underperform, rather than that of standardization, or reducing the #s based off variation instead. We attempted to do so primarily on the independent Y variable, and the method we chose was to reduce the state revenue to a number between 0 and 1 (the hypothesis of normalization) based on the number of students for that given school- unfortunately, this has the added benefit of combating outliers (which is a serious problem in our data) which standardization does not have. Our team had to shelve this idea temporarily due to the aforementioned problems faced with our dataset (requiring an unrealistic/impossible amount of cleaning). Our model as it stands is flawed and will underperform on our data because of this lack. 

Provide documentation for integration 

No readMe.MD is required for this implementation as it is a very simple model- but in future revisions we will want to have a readME.MD documentation to help the user understand how to utilize the given files. 

### Tools and Infrastructure Tried and Not Used

Describe any tools and infrastructure that you tried and ended up not using. What was the problem? Describe infrastructure used. 

One method our group attempted to utilize was KNN regression/classification. We initially thought it a good method due to its diverse capabilities, but unfortunately it fell short when it came to the high dimensionality of our data. To fix this we were able to reduce the data to what we decided were more pertinent features (manually, not via methods such as Principal Component Analysis) to a more manageable size such as illustrated in the homework assignments. Despite this, due to the problems described throughout this report our team decided the best route to take would be a logistical approach as our baseline. However, this model is not without its problems, the primary concern brought up by Al was that logistical classification falls short when exposed to extremely large data- and given that enormous quantities of data are standard in this report, that is a massive concern. 

## Experiments

Give a detailed summary of the results of your work.

 * Setup - Here is where you specify the exact performance measures you used.  
   * Describe the data used in experiment for presenting dataset: Datasheets for Dataset template 
   * Describe your accuracy or quality measure, and your performance (runtime or throughput) measure. 
   
 * Please use visualizations whenever possible. Include links to interactive visualizations if you built them. 
 
 * You can also submit a separated notebook as an appendix to your report if that makes the visualization/interaction task easier. 
   * It would be reasonable to submit your report as a notebook, but please make sure it runs on one of the two standard environments, and that you include any required files. 

## Conclusion
In this section give a high-level summary of your results. If the reader only reads one section of the report, this one should be it, and it should be self-contained. You can refer back to the Experiments Section for elaborations. This section should be less than a page. In particular emphasize any results that were surprising. 

This report details our team’s goal to examine a large dataset of [mostly] independent features and create a qualitative prediction of the school’s percent chance of falling under what would be deemed the ‘poverty level’ (which itself is a rather controversial/not-concrete number, up to debate unlike the baseline poverty for standard of living). We decided that the best route to go about achieving this goal would be a classification-based model- our group came to this conclusion because such approaches excel at outputting a value between 0 and 1 to give you a % value of your prediction. In other words, what we want to do is go about examining a range features, and then coming to the conclusion of the likelihood of the object being deemed an under-funded- which would be exactly what we want to do, examine the features of a school and determine the related financial ranking. 

In order to achieve this classification-based metric our group initially decided that a KNN based approach would perform best for this given dataset, but later turned to utilize a logistical approach instead. We came to this decision after conversing about the pros and cons of the two approaches and which would be best in what scenario. After a long discussion about the ideas and concepts discussed in the midterm our team concluded that while KNN can be quite effective and especially can dominate in cases that the ‘decision-boundary’ is moderately non-linear— but the reason we chose against it is because of the lack of customization (being that it has no parameters besides K) and the lack of principal component analysis, or in other words, the ability to determine if a given feature is pertinent. The benefits of logistical classification were not as apparent, and given that LDA also has many downsides: for instance assuming a normal distribution and common variance between features- however it is able to outperform linear classification (LDA) due to the nonlinear distribution of our data. Paired with the information gained from professor Tesic in lecture (that focusing purely on linear regression is ‘good enough’ for this midterm-check-in) our group decided that focusing on a logistical approach would be a good baseline model (or potentially a half-decent one in comparison to future analyzation of other methods) and it would be capable of outperform linear approaches at the same time! 

Our team’s first step in using logistical classification was of course to decide which of the features we would want to choose as our ‘y’ value- this challenge was much simpler in the homeworks as we were usually given an obvious red-herring feature in a clear, discrete range such as 1,2,3,4 or 0,1. However, in this CCD dataset we are working with for this report is much higher-level and lacks some of the ‘ease-of-access' features found in the homework. Reaching the decision of what will become our independent feature was easier said than done though, our group had many different opinions; primarily that a number purely based or ripped from the features directly would give a poor prediction of our model’s functionality- and thought to improve this by creating a new feature such as demonstrated in the homework. To do so our goal was to obtain a new number: revenue divided by the number of students, to help normalize our data and give each data-point a more average weight, so that a school with 10 students would not be unfairly pitted against that of a 1000 student school. Unfortunately, we could not find a method of achieving this at this time, due to the CCD’s data being deemed ‘unavailable’ for the related features such as population or similar metrics- and so we opted to create a range-based ‘independent’ y column, so if a school had a below average revenue it would be rated as a 0 and otherwise a 1. This of course falls victim to the weighting of schools not being the same and is in dire need of standardization to prevent unwanted ‘skewing’ of the data, and potentially causes our model to underperform temporarily but functions as a baseline model for analysis as discussed earlier on in the semester during the time of homework 3. 

At this point in the assignment our group had obtained our test and training data split with an 80/20 ratio which was deemed in lecture to be the most effective route to take by default, which also has the added benefit of allowing us to utilize a full range of machine learning techniques. Once our group obtained our test and training data the next step in the machine learning pipeline was of course to fit the data to combat the bias-variance trade-off and from that we could predict a potential imaginary value, also known as our probability! This process is quite computationally intensive manually, as computing each and every needed coefficient for the various logistical child-functions is an hour-long process for our group; but is beautifully encapsulated using the python sklearn library into a two line procedure. The end result of this logistical algorithm was to create a pair plot that would allow us to examine the relation or correlation of our variables, for instance: if revenue is high then we can qualitatively say the school-lunch-program would tend to also be high, and it is with this information that our group can create a model to predict, if given X, Y and Z that are correlated as described above, then our model has a N% probability of falling in the range of what would be deemed poverty. 

## References
List the references that cited in your project. 

https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/ 

https://books.google.com/books?hl=en&lr=&id=64JYAwAAQBAJ&oi=fnd&pg=PR13&dq=logistic+regression+versus&ots=DteR4V5onL&sig=MBA1PuQa9v-cFqga-9nOVWAtmqk#v=onepage&q=logistic%20regression%20versus&f=false 

## Appendix## 

Explain the contributions of each member to the project. Include all supporting materials, e.g., additional figures/tables, Python code technical derivations.
