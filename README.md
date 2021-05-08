# Bayesian Linear Regression

A simple Library built on top of tensorflow.js which allow you to solve bayesian linear regression problems.

<div style="text-align: center">
    <img src="assets/darkBLR.png" width="800px"/>
</div>

🗒: Note that this project is meant to be used for learning and researching purposes only and **it is not meant to be used for production.**

## Setup

Download or clone this repo and then add this to your index.html

```html

<script src="your_file_path_here/lib/BLR.js"></script>
```


## Usage

```javascript

/* generating synthetic data */
const x = tf.linspace(0.1,1.0, 5).expandDims(1); 
const y = tf.sin(x);
const test_x = tf.linspace(-1.5, 1.5, 10).expandDims(1);
const test_y = tf.sin(test_x);

/*specify hyperparameters */
const alpha = 5e-3; // parameter variance
const beta  = 11.1; // predictive variance 

/* creating a model and predicting y */
const blr = BLR(x,y,test_x,test_y);

// fitting a polynomial kernel of degree 8. and fetching our predicted y'es.
const { y:predictedY , yVariance : predVariance } = blr.useBasisFn("polynomial",{degree: 8}).train(alpha,beta).test();


predictedY.print() 
predVariance.print()

```

**NOTE**: if alpha and beta are not specified, the function will automatically try to learn the hyperparameters from the training data by maximizing the model evidence.

but unfortunatly the huge drawback of this method is that they don't always converges. mostly due to the fact that, it rely heavily on the initial alpha and beta so, if it isn't working for you, then you might have to try again by setting different init values of alpha and beta :

```javascript

// init hyperparams
const alpha =  0.01;
const beta  =  1e-5;

// finding the best alpha and beta 
const {alpha : newAlpha , beta : newBeta } = blr.evidenceMaximization(initAlpha = alpha,initBeta = beta);
```

#### Model-selection

We can also do model selection like for eg. if we want to find the most optimial degree for our polynomial function or if we want to find weather should i use gaussian or my own custom basis function.

We can do that by using evidenceFn() method, which just calculate the marginal-log-likelihood and any model which has higher value will be our best model.

```javascript

const results = [];
for(let i=0;i<1;i++){
    blr.useBasisFn("polynomial",{degree: i}).train();
    let k = blr.evidenceFn(trainX,trainY,alpha,beta).flatten().arraySync();
    results.push(k)
}

```

Now if we plot the results it looks something like this:-

<img src="assets/plot.png" width="500px"/>

which clearly suggest that for this data, degree 3 polynomial is the most optimal option for us.

#### Generating Data

Because we have learned our mean and variance of our parameter distribution we can easily generate pedicted Y by sampling weights from parameter dist using genY() method.

```javascript

/* ...some code... */

let blr = BLR(x,y,test_x,test_y);
blr.useBasisFn("polynomial",{degree: 6}).train();

// generate 10 new curves by using the weights sampled from parameter distribution.
blr.genY(SampleSize = 10);

```

#### Custom Basis Function

we can also use different basis function as well such as 
gaussian, identity basis functions. or even pass our own custom basis function.

```javascript

// fit 10 gaussian functions onto our data sets. 
blr.useBasisFn("gaussian",{nGaussians: 10});

// identity basis function which is juts f(x) = x;
blr.useBasisFn("identity");

// creating our custom basis function
const myBF = function(x,param = {pow : 2}){
    const pow = param;

    return tf.pow(x,pow)
}

// adding our basis function and using it to fit our data.
blr.addBasisFn(myBF , "myBasisFn").useBasisFn("myBasisFn",{pow: 5});

```
## License
[MIT](https://choosealicense.com/licenses/mit/)
