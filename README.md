# Predicting Conflict

UPDATE 2024/2/12: I do not have the free time I want to dedicate to this project, but I plan to pick it back up when I find the time. 

Predicting conflict using a variety of economic and socioeconomic factors.

## Project Definition

### Purpose and Scope

The purpose of this project is to build a machine learning model that can predict when a future conflict might arise based on certain economic and socioeconomic indicators.

### Definition of Conflict

For the purposes of this project, "conflict" will be defined as an armed confrontation between nation-states, or between a nation-state and non-state actors, that results in significant casualties or political impact. This definition will include wars such as WW1, WW2, Vietnam, Korea, as well as military interventions such as the Gulf War, Afghanistan War, or Iraq War. Additionally, political and social unrest may be considered as potential indicators of conflict, as these can lead to military action or international tensions. Please note that this definition may be refined or adjusted as the project progresses.

### Expected Outcome

The expected outcome of the project is a machine learning model that can accurately predict the likelihood of a conflict based on the selected economic and socioeconomic indicators.

### Previous Research

I will be reading through these and getting my foundational theories from them. Articles listed in bold sound most applicable based on their titles.

- **Bagozzi, B. E. (2015). Forecasting civil conflict with zero-inflated count models. Civil Wars, 17(1), 1-24.** 
    - Talks about what's called a "count model", which is used to analyze data about events that occur during conflicts, they identify 2 limitations of count models: not clear whether a "zero-inflated" count model is better than other, less complex count models, and that there is a need for an effective way to evaluate the accuracy of count models in predicting events which is important, to address these they used a new approach to evaluating the predictive accuracy of count models and demonstrate that zero-count are better than other models
    - Useful if we want to do a count-model, and specifically using their new approach at evaluating accuracy for using a zero-count
- **Beger, A., Morgan, R. K., & Ward, M. D. (2021). Reassessing the role of theory and machine learning in forecasting civil conflict. Journal of Conflict Resolution, 65(7-8), 1405-1426.**
    - Examine the protocols in "Forecasting civil wars: Theory and structure in an age of “Big Data” and machine learning", critique their theory-based approach and talk about how the findings in that rely completely on the use of an accuracy calculation called parametrically smoothed ROC curves, when the mistakes are corrected (and standard empirical ROC curves are used) they find that the claim that article makes is incorrect, instead they suggest that predictive modeling and ML should try to strengthen previous models regardless of whether they are theory-based or not
    - If we decide to use the theory based approach, we need to read this to see if we agree that smoothed ROC curves are not a good accuracy calculation
- **Blair, R. A., & Sambanis, N. (2020). Forecasting civil wars: Theory and structure in an age of “Big Data” and machine learning. Journal of Conflict Resolution, 64(10), 1885-1915.**
    - Use a simple model with as few variables as possible to avoid overfit, grounded in "contentious politics" theory, effective model (possibly more effective than other models using more standard, data-driven or structural variables)
    - Probably worth a good read if we decide that the critique of it is not founded
- **Cederman, L. E., & Weidmann, N. B. (2017). Predicting armed conflict: Time to adjust our expectations?. Science, 355(6324), 474-476.**
    - Overview of current conflict prediction using big data and machine learning techniques, highlights the strengths and limitations of existing prediction models (recent advances in temporal and spatial disaggregation), has different types of predictors used such as structural variables and war-related news reports, examples of country-level and civil conflict predictions using different types of models
    - Despite being a magazine? article, actually seems like it has a lot of info on our topics, gives an overview rather than trying to defend their own work
- **Colaresi, M., & Mahmood, Z. (2017). Do the robot: Lessons from machine learning to improve conflict forecasting. Journal of Peace Research, 54(2), 193-214.**
    - Provides framework for evaluating "out-of-sample" forecasts in conflict studies and improving practical performance of existing models, highlights the iterative tasks of "Box's loop" (build, compute, critique, and think), mentions the underutilized process of model criticism, talks about a software they made for new visualizations that build upon already existing tools, aim to help accelerate innovations across conflict studies and develop more effective models for conflict prediction/prevention
    - Useful, look into box's loop, figure out what kind of software they made bc it sounds handy
- **D’Orazio, V., Honaker, J., Prasady, R., & Shoemate, M. (2019, December). Modeling and forecasting armed conflict: AutoML with human-guided machine learning. In 2019 IEEE International Conference on Big Data (Big Data) (pp. 4714-4723). IEEE.**
    - Discusses the potential of AutoML and HGML in conflict research, they say AutoML can improve model selection and assessment in predictive models of conflict even without ML expertise, HGML offers options to customize for specific questions conflict researchers have, examined three papers with predictive models of conflict and found their HGML system using AutoML engines produced elevated performance on each model
    - Useful, reviews 3 other papers and shows that autoML can be used for easy benefits and we maybe can use
- **Hegre, H., Buhaug, H., Calvin, K. V., Nordkvelle, J., Waldhoff, S. T., & Gilmore, E. (2016). Forecasting civil conflict along the shared socioeconomic pathways. Environmental Research Letters, 11(5), 054002.**
    - Statistical model of 1960-2013 using effect of key socioeconomic variable on countr-specific conflict indicence, then forecasted 2014-2100 using their model along the 5 SSPs (new scenario data, shared socioeconomic pathways), they found that the scenario that led to sustainability and low climate change also led to global peace, then suggested that bringing up poor countries is more effective than more improvements to wealthy countries
    - Not sure if useful, talks about the poors
- **Hegre, H., Karlsen, J., Nygård, H. M., Strand, H., & Urdal, H. (2013). Predicting armed conflict, 2010–2050. International Studies Quarterly, 57(2), 250-270.**
    - Dynamic multinomial logit model estimation on a 1970-2009 cross-sectional data set of changes between no armed conflict, minor conflict, and major conflict, core predictors are population size, infant mortality, demographic composition, education levels, oil dependence, ethnic cleavages, and neighborhood characteristics, predictions by simulating behavior of the conflict variable using projections for predictors from UN World Population Prospects and the International Institute for Applied Systems Analysis, predicted well with true positive rate of 0.79 and false positive of 0.085
    - Useful, seems actually very detailed and uses a diff model than I've seen so far, sounds interesting
- **Hegre, H., Nygård, H. M., & Landsverk, P. (2021). Can we predict armed conflict? How the first 9 years of published forecasts stand up to reality. International Studies Quarterly, 65(3), 660-668.**
    - Actually reviews the previous article (Predicting armed conflict, 2010–2050) using multiple metrics, they found that it was able to make meaningful and reasonably accurate predictions, found that it was better at large-scale conflict rather than low-level, missed a couple regional shifts
    - Useful in conjunction with the one it reviews, provides guidance for further research (us)
- **Perry, C. (2013). Machine learning and conflict prediction: a use case. Stability: International Journal of Security and Development, 2(3), 56.**
    - Seems actually like a user-guide on how to use ML to predict conflict, talks about how the UN has been working on models, how ML is promising, discuss different methodologies, claim that ML provides better predictive power than using prior violence (which I assume was a standard), suggestions for improvements
    - Very useful, seems like we can use this to get a good baseline knowledge of how to use ML for this use case
- **Weidmann, N. B., & Ward, M. D. (2010). Predicting conflict in space and time. Journal of Conflict Resolution, 54(6), 883-901.**
    - Basically using geography as another predictor in their spatially and temporally autoregressive discrete regression model, following other author's framework, model applied to geo-located data on attributes and conflict events in Bosnia over 1992-95, they found strong relation for spatial/temporal dimension
    - Kind of useful, if we want to include geography, which they say improves over normal regression which only has time lag, worth a look
- Blattman, C., & Miguel, E. (2010). Civil war. Journal of Economic literature, 48(1), 3-57.
- Collier, P. (2000). Economic causes of civil conflict and their implications for policy.
- Cramer, C. (2003). Does inequality cause conflict?. Journal of International Development: The Journal of the Development Studies Association, 15(4), 397-412.
- Gilligan, M. J., Pasquale, B. J., & Samii, C. (2014). Civil war and social cohesion: Lab‐in‐the‐field evidence from Nepal. American Journal of Political Science, 58(3), 604-619.
- Guo, W., Gleditsch, K., & Wilson, A. (2018). Retool AI to forecast and limit wars.
- Hirose, K., Imai, K., & Lyall, J. (2017). Can civilian attitudes predict insurgent violence? Ideology and insurgent tactical choice in civil war. Journal of peace research, 54(1), 47-63.
- Nielsen, R. A., Findley, M. G., Davis, Z. S., Candland, T., & Nielson, D. L. (2011). Foreign aid shocks as a cause of violent armed conflict. American Journal of Political Science, 55(2), 219-232.
- Østby, G. (2008). Inequalities, the political environment and civil conflict: Evidence from 55 developing countries (pp. 136-159). Palgrave Macmillan UK.
- Theisen, O. M., Gleditsch, N. P., & Buhaug, H. (2013). Is climate change a driver of armed conflict?. Climatic change, 117, 613-625.
- Thorbecke, E., & Charumilind, C. (2002). Economic inequality and its socioeconomic impact. World development, 30(9), 1477-1495.
- Weidmann, N., & Ward, M. D. (2008, September). Spatial–temporal modeling of civil war: The example of Bosnia. In GROW-Net Conference, Zurich.

## Data Collection

### Data Sources

The data sources that will be used for this project will be identified based on previous research and domain expertise. These may include, but are not limited to:

- The World Bank: The World Bank provides a wide range of data on economic and social development, including indicators such as GDP, poverty rates, education levels, and health outcomes. The World Development Indicators (WDI) and the Global Economic Monitor (GEM) are two such databases.
- The United Nations: The United Nations (UN) provides data on a variety of economic and social issues through its various specialized agencies and programs, such as the UN Development Programme (UNDP), the UN Statistics Division, and the International Labour Organization (ILO). The UN also publishes reports on topics such as human development, gender equality, and sustainable development.
- International Monetary Fund (IMF): IMF provides data on economic indicators, including GDP, inflation rates, and trade flows, from countries around the world.
- Transparency International: The Corruption Perceptions Index by Transparency International provides data on corruption levels in countries around the world.
- United Nations High Commissioner for Refugees (UNHCR): The UNHCR provides data on forced displacement, including refugees and internally displaced persons.
- Human Rights Watch: Human Rights Watch is an international non-governmental organization that provides data on human rights violations around the world.
- International Crisis Group: International Crisis Group provides analysis and data on conflicts and potential conflicts around the world.
- Armed Conflict Location and Event Data Project (ACLED): ACLED provides data on conflict events, actors, and their attributes for countries in Africa, South Asia, and the Middle East.
- Social Progress Index: The Social Progress Index is a global index that measures the social and environmental outcomes of countries.

### Data Collection and Cleaning

### Indicators for Conflict Prediction

The specific economic and socioeconomic indicators that will be used to predict conflicts will be identified based on previous research and domain expertise. These indicators may include, but are not limited to:

#### Economic Indicators

- Gross domestic product (GDP)
- Income inequality
- Unemployment rate
- Poverty rate
- Stock market indices (e.g. Dow Jones Industrial Average, S&P 500)
- Consumer Price Index (CPI)
- Producer Price Index (PPI)
- Trade balance (exports minus imports)
- Federal funds rate (interest rate set by the Federal Reserve)
- National debt and budget deficit
- Real estate prices
- Crude oil production
- Crude oil imports and exports
- Strategic Petroleum Reserve (SPR) levels
- Oil consumption and demand
- Gasoline prices
- Dependence on primary commodity exports

#### Socioeconomic Indicators

- Education level
- Access to healthcare
- Crime rate
- Political instability
- Military expenditure
- Ethnic and religious tensions
- Infant mortality rate
- Primary school completion rate
- Political rights
- Ethnic fractionalization
- Low education levels
- Weak state institutions
- Measures of social cohesion
- Natural resource availability
- Human rights violations

#### Business and Industry Indicators

- Consumer and business confidence indices
- Manufacturing and services sector indices

Keep in mind that some of these may be highly correlated, and are only initial thoughts as to what indicators might be included. These have not been based on prior research as of writing (2023-03-25).

### Exploratory Data Analysis

### Data Storage and Management

## Feature Engineering

### Feature Selection

### Feature Creation

### Feature Normalization and Scaling

## Model Selection

### Algorithm Selection

### Model Training and Testing

### Hyperparameter Tuning

## Model Evaluation and Interpretation

### Performance Evaluation

### Sensitivity Analysis

### Model Interpretation

### Result Communication

## Deployment

### Implementation

The trained model will be implemented in a production environment to make predictions on new data. The implementation process will involve integrating the model with other relevant systems and software.

### Documentation and Instructions

Documentation and instructions will be provided to explain how to use the model and interpret its results. The documentation will be clear and concise, and will provide technical and non-technical users with the necessary information to use the model.

### Performance Monitoring

The performance of the model will be monitored in a production environment to ensure that it is working as intended. Any issues or errors will be addressed promptly to ensure that the model's predictions are reliable.

### Model Improvements

The model will be updated and improved based on feedback and performance monitoring results. Improvements may include changes to the model's algorithms, features, or data sources.

### Maintenance and Support

Ongoing maintenance and support will be provided to ensure that the model remains up-to-date and continues to provide accurate predictions. Regular reviews and updates will be conducted to address any issues or challenges that arise.

## Ethics and Privacy

### Ethical Considerations

The use of data and development of the model will be conducted in an ethical and responsible manner. Relevant laws and regulations will be followed to ensure that the project is compliant with ethical and legal standards.

### Bias Considerations

Potential biases in the data and model will be considered and addressed to ensure that the predictions are fair and unbiased. Different methods of identifying and addressing bias will be explored and implemented.

### Transparency and Accountability

The author will be transparent and accountable in their use of data and development of the model. Clear communication channels will be established to ensure that stakeholders are informed about the project's progress and results, and to address any concerns or questions that arise.

## Conclusion

### Project Outcomes

The outcomes of the project will be summarized and presented. This may include a report or presentation that highlights the project's objectives, methodology, results, and insights.

### Future Research and Improvement

Areas for future research and improvement will be identified based on the project's outcomes and insights. This may include exploring different machine learning algorithms, features, or data sources, or conducting further analysis to address specific research questions or challenges.

### Lessons Learned

Lessons learned from the project will be documented and shared with the project team and relevant stakeholders. This will ensure that future projects can benefit from the project's successes and challenges.
