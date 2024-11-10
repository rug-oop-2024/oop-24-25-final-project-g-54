
DSC-0001: Implementation of an Abstract Base Class for Model Interface
======================================================================

**Date:** 

2024-11-06

**Decision:** 

Defining the base Model class as an Abstrac base Class (ABC)

**Status:** 

Accepted

**Motivation:** 

Having a blueprint where other classes can inherit from, where an implementation of a fit and predict methods is forced. 

**Reason:** 

To promote consistency across all the classes that inherit Model.

**Limitations:**

Requires each class to override methods from the parent class.

**Alternatives:**

Using a normal class without enforced methods overrides.


DSC-0002:  Implementation of LassoCV
====================================

**Date:**

2024-11-09

**Decision:**

Implementation of LassoCV rather than the normal Lasso. 

**Status:**

Accepted

**Motivation:**

LassoCV automates alpha selection with cross-validation improving performance by optimizing regularization.

**Reason:**

LassoCV finds the best alpha automatically reducing manual tuning and enhancing accuracy.

**Limitations:**

Higher computation time.

**Alternatives:**

Use standard Lasso with a fixed alpha.


DSC-0003:  Encapsulation of model attributes
============================================

**Date:**

2024-11-06

**Decision:**

Implementation of getters for Encapsulating model attributes

**Status:**

Accepted

**Motivation:**

To prevent unintended modifications maintaining data integrity of the model's attributes such as type and parameters

**eason:**

Getter methods ensure controlled access to parameters which are returned as copies to avoid side effects.

**Limitations:**

Requires subclasses to implement abstract methods. 

**Alternatives:**

Use of public attributes.


DSC-0004:  Implementation of KNN as a classification model 
==========================================================

**Date:**

2024-11-06

**Decision:**

Implementation of KNN as a classification model 

**Status:**

Accepted

**Motivation:**

KNN is simple, interpretable and works well on various classification tasks making it a good addition.

**Reason:**

KNN uses similarity to classify data making it adaptable to different datasets without complex tuning.

**Limitations:**

KNN can be slow on large datasets as it calculates distances for each prediction.

**Alternatives:**

Other models like decision trees.


DSC-0005:  Implementation of Neural Network (MLP) classifier
============================================================

**Date:**

2024-11-06

**Decision:**

Implementation of MLP classification model.  

**Status:**

Accepted

**Motivation:**

MLP can capture complex patterns, enhancing the classification model.

**Reason:**

MLP’s ability to handle non-linear relationships makes it suitable for complex datasets and complements simpler models.

**Limitations:**

MLP requires more resources and tuning than simpler models.

**Alternatives:**

Use simpler models like KNN.


DSC-0006:  Implementation of random_forest classifier
=====================================================

**Date:**

2024-11-06

**Decision:**

Implementation of Random Forest Classifier

**Status:**

Accepted

**Motivation:**

The Random Forest algorithm is robust and effective for various classification tasks

**Reason:**

Combining multiple trees helps capture complex patterns and improves generalization.

**Limitations:**

Random Forest models can be slower for large datasets and require more memory.

**Alternatives:**

Use simpler models like KNN.



DSC-0007:  Implementation of multi-linear regression
====================================================

**Date:**

2024-11-06

**Decision:**

Implement Multiple Linear Regression as a regression model.

**Status:**

Accepted

**Motivation:**

Linear regression is simple, interpretable and effective for predicting continuous values.

**Reason:**

Fits a linear equation to capture relationships between multiple features and outcomes.

**Limitations:**

Limited to linear relationships which may not suit complex data.

**Alternatives:**

Use more complex models.



DSC-0008:  Implementation of gradient_bostingR
==============================================

**Date:**

2024-11-06

**Decision:**

Implementation of gradient_bostingR as regression model.

**Status:**

Accepted

**Motivation:**

Gradient boosting improves accuracy by combining multiple decision trees.

**Reason:**

Minimizes prediction error making it effective for complex data.

**Limitations:**

Higher computation time.

**Alternatives:**

The use of a simpler regression model.



DSC-0009: NotFoundError for Missing Paths
=========================================

**Date:**

2024-11-06

**Decision:**

Implementation of a NotFoundError.

**Motivation:**

Loading datasets and artifacts requires specific error handling for missing files.

**Reason:** 

NotFoundError provides clear messaging when an artifact path is missing improving reliability in AutoMLSystem and storage operations.

**Limitations:**

Only handles path-related errors.

**Alternatives:** 

Use FileNotFoundError.

