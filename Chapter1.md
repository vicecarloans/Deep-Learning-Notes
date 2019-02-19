## I. AI vs ML vs Deep Learning
![comparison](./comparison.png)

**1. AI:**
    + Chess Player
    + Hard-coded rules and algorithms
    + Encompass all ML and DL
    
**2. Machine Learning:**
    + Trained based on inputs, examples
    + Tightly related to statistics
    + Dealing with large complex datasets
    + 3 main things: Input, Examples of output, a way to measure algorithms
    + Automatic search process for better representation
    + No creativity in finding transformations but only searching through a predined set of operations, called a *hypothesis space*
    + Mapping inputs to targets done by observing examples of inputs and targets
    
**3. Deep Learning:**
    + Using *layers* to inrease meaningful representations
    + *deep* prefers to *successive layers* of representations
    + Using *weights* to map into layers
    + If there's a loss, it will use Optimizer and reevaluates the weights again

## II. Advantage of Deep Learning:
    
    
   + Make problem solving is easier because it automates feature engineering.
   + Two characteristics of how deep-learning learns data:
       + **Incremental** : layer-by-layer in which increasingly complex representations are developed. 
       + They are learned **jointly**
    + *gradient propagation*: better *activation functions* for neural layers; better *weight-initialization schemes*; better *optimization schemes* such as RMSProp and Adam 