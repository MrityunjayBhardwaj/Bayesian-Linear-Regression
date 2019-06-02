(
    // TODO: Add a parameter distribution DONE!
    // TODO: Add predictive distribution. DONE!
    // TODO: Add Evidence Approximation   DONE!

function(global){

    /**
     * @summary initializing Linear Basis Function .
     * @description NOTE: Inputs must be a tf.tensor Object
     * 
     * @param {object} trainX  
     * @param {object} trainY 
     * @param {object} testX 
     * @param {object} testY 
     */
    function LBF(trainX = [],trainY = [],testX = [],testY = []){

        return new lbf(trainX,trainY,testX,testY);
    }

    function lbf(trainX,trainY,testX,testY){

        let basisFnNames = ["identity","polynomial","gaussian"];

        // this array contains all of the available basis functions.
        let basisFnArr = [
            /* Identity function: */ ((x)=>x), 

            function Polynomial(x,fnParams = {degree: 5}){
                /* polynomial function */
                // ? should developer be concerned about the dimensions?? i think yes. we will add it later;

                // console.log("this workis y: ",x.shape);
                const {degree: degree} = fnParams;
                const polyVec  = tf.linspace(0, degree, degree+1).expandDims(1); // array of all the powers 02Degree

                y = x.pow(polyVec.transpose());

                // console.log(tf.isNaN(x),x,tf.zeros(x.shape));
                /* here we are cleaning our y by replacing all the NaNs by zeros */
                return y.where(tf.isNaN(y),y,tf.zeros(y.shape));
            },
            function Gaussian(x,fnParams = {nGaussians : null , mean: null,variance:null}){

                // by default mean is calculated by sub-dividing our training data range and take it as a mean.
                const mu = fnParams.mean || tf.linspace(trainX.min().flatten().arraySync()[0],
                                       trainX.max().flatten().arraySync()[0],
                                       fnParams.nGaussians)
                                       .flatten().arraySync();

                const sigma = fnParams.variance || tf.scalar (0.1);


                if (fnParams.nGaussians !== mu.shape[0]){
                    throw new Error("invalid Input : Miss-match in given mean and nGaussians");
                }

                y = tf.ones(x.shape);

                for(let i=0;i<fnParams.nGaussians;i++) {

                // exp(-(x-mu)**2)/sigma)
                k = tf.exp( tf.mul(tf.scalar(-1),
                                   tf.div(
                                          tf.pow( tf.sub( x ,tf.scalar( mu[i] ) ) , 2 ),
                                          tf.mul(tf.scalar(2),tf.pow(sigma,2))
                                          )
                                    )
                            )
                y = y.concat(k
                ,axis=1);

            }
                return y;
            }
        ];

        // current basis function in use.
        let cBasisFn = function(x){return x;};
        let cBasisFnParams = {};

        // this model object contains everything we need in order to estimate our outputs.
        let model = {
            Weights: {mean:null,variance:null},
            hyperParameters: {alpha : 0.001,beta: 0.00001}
        };


        /* getters and setters for private vars:- */

        /**
         * @summary set our training x and y.
         * @description NOTE: Input must be a tf.tensor Object
         * @param {object} newTrainX
         * @param {object} newTrainY
         */
        this.setTrainingData = function(newTrainX,newTrainY){
            trainX = newTrainX || trainX;
            trainY = newTrainY || trainY;

            return this;
        };
        this.getTrainingData = function(){
            return {trainX: trainX,trainY: trainY};
         };

        /**
         * @summary set our testing data; x and y.
         * @description NOTE: Input must be a tf.tensor Object
         * @param {object} newTestX
         * @param {object} newTestY
         */
        this.setTestingData = function(newTestX,newTestY){
            testX = newTestX || testX;
            testY = newTestY || testY;

            return this;
        };

        this.getTestData = function(){
            return {testX: testX,testY: testY};
        };

        /**
         * @summary specify which Basis function to use.
         * @description it takes 2 arguments (1): function name and (2): params, each params 
         * are unique to there respected basis function.
         * by default there are 3 built in basis functions:-
         *  - "Identity" : which is simply f(x) = x; takes no special parameters.
         *  - "polynomial" : fits a polynomial curve of specific degree specified.
         *  - "gaussian" : fits multiple number of gaussian function 
         * if the second argument isn't specified then all the parameters are learned 
         * from the data using model selection.
         * @example 
         * this.useBasisFn("identity");
         * 
         * @example 
         * // basis fn of a cubic polynomial
         * this.useBasisFn("polynomial",{degrees: 3});
         * 
         * @example 
         * // basis function of 4 gaussians having specified mean (and can also specify variance.)
         * this.useBasisFn("gaussian",{nGaussians : 4,mean: tf.tensor([0.1,0.2,0.4,0.8]) });
         * @param {string} fnName specify name of the basis fn which you want to use.
         * @param {object} params An object of parameters which gets feeded to our selected basis function.
         */
        this.useBasisFn = function(fnName = "polynomial",params){
            if (basisFnNames.indexOf(fnName)){
                // if its our default function then use that.

                let fnIndex = basisFnNames.indexOf(fnName);
                cBasisFn = basisFnArr[fnIndex];

                if(typeof params === "object")
                    cBasisFnParams = params;
                // console.log(cBasisFn,cBasisFnParams);
            }

            return this;

        };

        /**
         * @summary you can also add a custom basis function using this function.
         * @param {function} newFn new basis function to add.
         * @param {string} name name of this new basis function
         */
        this.addBasisFn = function(newFn,name="newFunction"){

            // TODO: Validate the function and its output dimensionality.

            let count = 0;
            while(basisFnNames(name)){
                // if the name is already in use chage it.
                name = name+`_${count}`
                count++;
            }

            basisFnNames.push(name);
            basisFnArr.push(newFn);
        };

        this.getBasisFnNameList = function(){
            // gives the names of all the basis function.
            return basisFnNames;
        };

        /**
         * @summary this function calculates the mean and variance of our weights for the given data.
         * @param { tf.tensor } x 
         * @param { tf.tensor } t 
         * @param { tf.tensor } alpha specify the uncertainty in our Weights 
         * @param { tf.tensor } beta  specify the uncertainty in our data
         * @return { object } {mean , variance , precision}
         */
        this.paramPDF = function(x,t,alpha,beta){
            /* calculates the pdf and sufficient statistics for our parameter "w" */

            const phiX = cBasisFn(x,cBasisFnParams);
           let precision_p1 = tf.mul(tf.scalar(alpha),tf.eye(phiX.shape[1])) // covariance matrix of isotropic gaussian.
           let precision_p2 = tf.mul(tf.scalar(beta),tf.matMul(phiX.transpose(),phiX));
           let precision    = tf.add(precision_p1,precision_p2);

           let variance = tf.tensor( math.inv(precision.arraySync()) ); // using gauss jordan method. 
           let mean = tf.mul(tf.scalar(beta) , tf.matMul(variance,tf.matMul(phiX.transpose(),t)));

           return { mean , variance , precision};
        };

        /**
         * @summary this function calculates our predicted y along with its uncertainty.
         * @param { tf.tensor } x 
         * @param { tf.tensor } weightMean mean of our Weight distribution for all the x'es
         * @param { tf.tensor } weightVariance variance in our Weights for all the x'es
         * @param { tf.tensor } beta specify the uncertainty in our data.
         * @return {object} {y , yVariance} .
         */
        this.predictedPDF = function(x,weightMean,weightVariance,beta){
            /* calculates the predicted Y and uncertianity associated with it. (result of using Bayesian Approach) */

            const phiX = cBasisFn(x,cBasisFnParams);
            y = phiX.matMul(weightMean);
            // uncertainty in our predictions.
            yVariance = tf.add (tf.scalar(1/beta) , tf.mul(tf.matMul(phiX,weightVariance),phiX).sum(1));

            return {y,yVariance};

        };

        /**
         * @summary computes evidence function a.k.a mariginal log likelihood function. for the given x'es
         * @param { tf.tensor } x 
         * @param { tf.tensor } y 
         * @param { number } alpha
         * @param { number } beta
         */
        this.evidenceFn = function(x,t,alpha,beta){
            /* compute the evidence function a.k.a log marginal likelihood */
            
            const weightMean  = model.Weights.mean;
            const weightVariance  = model.Weights.variance;
            const weightPrecision = tf.tensor(math.inv(weightVariance.arraySync()));


            const phiX = cBasisFn(x,cBasisFnParams);
            // const {mean:weightMean,variance:weightVariance,precision_} = this.paramPDF(phiX,t,alpha,beta);

            const {0:N , 1:M} = phiX.shape;
            const E_D = tf.mul( 
                                tf.scalar(beta) , 
                                tf.sum( 
                                        tf.pow( 
                                                tf.sub(
                                                        t ,
                                                        phiX.matMul(weightMean)
                                                      ),
                                                      2
                                               ),
                                       )
                              );

            const E_W  = tf.mul(
                                 tf.scalar(alpha) ,
                                 tf.sum(
                                        tf.matMul(
                                                weightMean.transpose() ,
                                                weightMean                                                )
                                        )
                                );
            
            const E_mN = tf.add(E_D , E_W);

            const eviFn_p1 = tf.scalar( M*Math.log(alpha) + N*Math.log(beta) - N*Math.log(2*Math.PI)) ;
            const eviFn_p2 = E_mN.neg();
            const eviFn_p3 = tf.log( math.det( weightPrecision.arraySync() ) ).neg();

            let eviFn =  tf.add( eviFn_p1,eviFn_p2 );
            eviFn     =  tf.add( eviFn,eviFn_p3 );
            eviFn     = eviFn.mul( tf.scalar(1/2) );

            return eviFn;
        }

        /**
         * @summary this function tries to find the alpha and beta by iteratively and jointly maximize the evidence function
         * @param { tf.tensor } x 
         * @param { tf.tensor } t 
         * @param { number } alpha_0 initial value of alpha.
         * @param { number } beta_0  initial value of beta.
         * @param { number } maxItr maximum number of iteration is allowed for this function to reach the maximum point.
         * @param { number } tollerance assume convergence if the gradient difference is smaller then this tollerance value. 
         * 
         */

        this.evidenceMaximization = function(x,t,alpha_0,beta_0,maxItr = 200,tollerance = 1e-2){
           
            alpha_0 = alpha_0 || 1e-3;
            beta_0  = beta_0  || 1e-5;

            const {0:N , 1:M} = x.shape;

            const phiX = cBasisFn(x,cBasisFnParams);
        //     let phiX = [[1, 0.001    , 0.000001, 0        , 0        ],
        //         [1, 0.112    , 0.012544, 0.0014049, 0.0001574],
        //         [1, 0.223    , 0.049729, 0.0110896, 0.002473 ],
        //         [1, 0.334    , 0.111556, 0.0372597, 0.0124447],
        //         [1, 0.445    , 0.198025, 0.0881211, 0.0392139],
        //         [1, 0.556    , 0.309136, 0.1718796, 0.0955651],
        //         [1, 0.667    , 0.444889, 0.296741 , 0.1979262],
        //         [1, 0.778    , 0.605284, 0.4709109, 0.3663687],
        //         [1, 0.8890001, 0.790321, 0.7025954, 0.6246073],
        //         [1, 1        , 1       , 1        , 1        ]]

        // t =   [[-0.0800581],
        //     [0.6038209 ],
        //     [1.0003519 ],
        //     [0.795733  ],
        //     [0.2109738 ],
        //     [-0.3361936],
        //     [-0.933997 ],
        //     [-0.9665811],
        //     [-0.5177265],
        //     [-0.0333505]]


            // t = tf.tensor(t);
            // phiX = tf.tensor(phiX);

            // phiX.print();


            // const trueEigen = [5.64487858e-05,4.82649380e-03,1.58017463e-01,2.55904474e+00, 1.65480440e+01] 

            eigenvalues_0 = (nd.la.eigenvals(nd.array(phiX.transpose().matMul(phiX).reverse().arraySync())));

            let foo = phiX.transpose().matMul(phiX).arraySync();
            // console.log(eigenvalues_0);
            // console.log(trueEigen);


            const eValsArr = [];
            for(let i=0;i<eigenvalues_0.shape[0];i++){
                eValsArr.push(eigenvalues_0(i).re);
            }

            eigenvalues_0 = tf.tensor(eValsArr)


            let alpha = alpha_0;
            let beta = beta_0;

            for(let i=0;i< maxItr ; i++){
                let beta_pre = beta;
                let alpha_pre = alpha;
                
                eigenvalues = tf.mul( eigenvalues_0 ,beta);

                const {mean: weightMean,variance: weightVariance,precision: weightPrecision} = this.paramPDF(phiX,t,alpha,beta);

                gamma = tf.sum(tf.div(eigenvalues, tf.add(eigenvalues , tf.scalar(alpha))));
                alpha = tf.div(gamma , tf.sum(tf.pow(weightMean, 2))).flatten().arraySync()[0] || alpha;

                beta =  tf.div( 
                                    1,
                                    tf.mul(
                                        tf.div(
                                                1,
                                                tf.sub(N,gamma)
                                                ),
                                        tf.sum(
                                                tf.pow(
                                                        tf.sub(
                                                                t,
                                                                tf.matMul(phiX,weightMean),
                                                            ),
                                                        2
                                                        )
                                                )
                                        )
                                    ).flatten().arraySync()[0]  || beta ;

                console.log(alpha,beta);
            // console.log("difference:",(Math.abs(alpha_pre - alpha)),(Math.abs(beta_pre - beta) ));
                if ((( (Math.abs(alpha_pre - alpha)) < tollerance) && (alpha_pre !== alpha ))
                                            &&
                    ( (Math.abs(beta_pre - beta) )  < tollerance) && (beta_pre !== beta ))
                {

                    console.log(`converged in ${i} iterations`);
                    return {alpha,beta};
                }
            }

            console.log("can't converge please try with different initial values");
            // return  this.evidenceMaximization(x,t,math.random()*1,0);
            return {alpha,beta};
        }

        /**
         * @summary Sample the Weights from parameter distribution. using learned mean and variance values.
         * @param { number } SampleSize number of samples you require.
         * @return { array } array of sampled weight values.
         */
        this.genW = function(SampleSize){
            /* generate Weights using inverse transform sampling */

            const weightArr = [];
            for(let i=0;i<SampleSize;i++){
                const cWeightSample = MultivariateNormal.default(
                                                                model.Weights.mean.flatten().arraySync(),
                                                                model.Weights.variance.arraySync()
                                                                ).sample();
                                                                
                weightArr.push(tf.tensor(cWeightSample).expandDims(1));
            }
    
            return weightArr; 
            
        };

        /**
         * @summary Sample y'es from the prediction distribution.
         * @param { number } SampleSize number of samples you need.
         * @return { array } array of sampled y values.
         */
        this.genY = function(SampleSize){
            // generate Y by first generating new Weight vectors from parameter distribution 
            // and then using tht to generate new Y.

            const newGenY = [];
            const phi_testX = cBasisFn(testX,cBasisFnParams);
            for(let weightVec of this.genW(SampleSize)){
                
                newGenY.push( tf.matMul(phi_testX,weightVec) );
            }
            return newGenY; 
        };

        /**
         * @summary this function trains our model using the given data and parameters
         * @description this function trains our model by first learn the alpha and beta
         * if none of them is provided/specified using evidenceMaximization,
         * then it uses these hyperparameters in learning our weights (the reason why we train our model.) 
         * then we use these weight to calculate our prediction. in test() function.
         * @param { number } alpha specify the uncertainty in our Weights 
         * @param { number } beta specify the uncertainty in our data
         */
        this.train = function(alpha=null ,beta = null){
            // in this block we are going to be calculating the hyperparameters aswell as the 
            // parameter distribution i.e, p(w | t,alpha,beta);

            

            if (!(alpha && beta)){
                // if the any one of the hyper parameter is unknown then approximate the values from the trianing data.
                const {alpha:newAlpha,beta:newBeta} = this.evidenceMaximization(trainX,trainY,alpha,beta);

                alpha = newAlpha || alpha || model.hyperParameters.alpha;
                beta  = newBeta  || beta  || model.hyperParameters.beta;

            }

            const {mean: weightMean,variance: weightVariance,precision: weightPrecision} =  this.paramPDF(trainX,trainY,alpha,beta);

            /* adding our parameters to our model. */
            model.Weights.mean = weightMean;
            model.Weights.variance = weightVariance;
            model.hyperParameters.alpha = alpha;
            model.hyperParameters.beta  = beta;

            return this;
        };

        /**
         * @summary this function produces our predicted "y"
         * @param { tf.tensor } newtestX if we want to test our model on a different dataset then what we have initially specified then we can specify this new testX here.
         */
        this.test = function(newtestX = null){

            // const phi_testX = cBasisFn( newtestX || testX,cBasisFnParams);
            return this.predictedPDF(( newtestX || testX),model.Weights.mean,model.Weights.variance,model.hyperParameters.beta);
        };

    }

    lbf.prototype = {
        /* parant function:- */

        // Maybe we can use the prototype of a potential Bayes.js Library.
    }

    /* assigning our function to our global Window Object */
    global.LBF = LBF;

    console.log("added LBF");

    }(window)
)


