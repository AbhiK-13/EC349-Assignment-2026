# EC349-Assignment-2026

This repository contains my assignment submission for the EC349 (Data Science for Economists) module, which required the processing and use of the AirBnB Listings dataset. There were two aspects of this task: a prediction task of annual occupancy (estimated_occupancy_l365d) and the causal estimate of superhost status (host_is_superhost). 

This report finds that random forest and bagging models best predict annual occupancy, where number of reviews and the minimum number of nights required for a booking are illustrated to be key drivers of prediction. An average of the estimates from the random forest and bagging models was taken to avoid obtaining an estimate determined mainly by noise. This is because, despite the two models having a similar R-sqaured and RMSE value, the MAE (Mean Absolute Error) for the bagging model was lower, indicating that it might be overfit and that its predictions were picking up noise as opposed to the true data-generating process. For median values of predictors included in the dataset, the annual occupancy is approxximately 49 days. 

Furthermore, superhost status is estimated to increase annual estimated occupancy by approximately 12 days across the year, having the strongest impact for hosts of private rooms in homes in Kingston upon Thames. For this estimate, the AIPW (Augmented Inverse Probability Estimator) was used over methods like naive OLS and metalearners as it is the most credible. However, its estimate is undermined by how the overlap and unconfoundedness assumptions needed for causal inference seem implausible in this context, as those with superhost status are inherently different from those without superhost status in unobservable variables (this violates unconfoundedness); a plot of the propensity score for being treated across both control and treatment units displays that there is a lack of comparable treatment units for control units with a lower propensity of being treated. 





