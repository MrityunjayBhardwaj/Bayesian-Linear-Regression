(
    // TODO: Add a parameter distribution DONE!
    // TODO: Add predictive distribution. DONE!
    // TODO: Add Evidence Approximation   In-Progress

function(global){

    function BCF(trainX = [],trainY = [],testX = [],testY = []){

        return new bcf(trainX,trainY,testX,testY);
    }

    function bcf(trainX,trainY,testX,testY){

        /* private properties: */
        // collection of all the bassis function name;
        let basisFnNames = ["identity","polynomial","gaussian"];
        let basisFnArr = [
            /* Identity function: */ ((x)=>x), 

            function Polynomial(x,fnParams = {degree: 5}){
                /* polynomial function */
                // ? should developer be concerned about the dimensions?? i think yes. we will add it later;

                console.log("this workis y: ",x.shape);
                const {degree: degree} = fnParams;
                const polyVec  = tf.linspace(0, degree, degree+1).expandDims(1); // array of all the powers 02Degree

                y = x.pow(polyVec.transpose());

                console.log(tf.isNaN(x),x,tf.zeros(x.shape));
                /* here we are cleaning our y by replacing all the NaNs by zeros */
                return y.where(tf.isNaN(y),y,tf.zeros(y.shape));
            },
            function Gaussian(x,fnParams = {nGaussians : null , mean: null,variance:null}){
                // TODO: Add a gaussian kernel.

                // calculating mean;

                // by default mean is calculated by sub-dividing the range and take it as a mean.
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
                // y = tf.ones(x.shape).concat(y,1);

                console.log("prinitng Gaussian Y:-",mu,sigma);
                y.print();
                return y;

            }
        ];

        // current basis function used.
        let cBasisFn = function(x){return x;};
        let cBasisFnParams = {};

        let Weights = {mean: null,variance: null};

        let model = {
            Weights: {mean:null,variance:null},
            hyperParameters: {alpha : null,beta: null}
        };

        /* public properties: */

        /* getters and setters for private vars:- */
        this.setTrainingData = function(newTrainX,newTrainY){
            trainX = newTrainX || trainX;
            trainY = newTrainY || trainY;

            return this;
        };

        this.getTrainingData = function(){
            return {trainX: trainX,trainY: trainY};
         };

        this.setTestingData = function(newTestX,newTestY){
            testX = newTestX || testX;
            testY = newTestY || testY;

            return this;
        };

        this.getTestData = function(){
            return {testX: testX,testY: testY};
        };

        // this.

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

        this.paramPDF = function(phiX,t,alpha,beta){
            /* calculates the pdf and sufficient statistics for our parameter "w" */

           let precision_p1 = tf.mul(tf.scalar(alpha),tf.eye(phiX.shape[1])) // covariance matrix of isotropic gaussian.
           let precision_p2 = tf.mul(tf.scalar(beta),tf.matMul(phiX.transpose(),phiX));
           let precision    = tf.add(precision_p1,precision_p2);

           let variance = tf.tensor( math.inv(precision.arraySync()) ); // using gauss jordan method. 
           let mean = tf.mul(tf.scalar(beta) , tf.matMul(variance,tf.matMul(phiX.transpose(),t)));

           return {precision, mean , variance};
        };

        this.predictedPDF = function(phi_testX,w_mean,w_variance,beta){
            /* calculates the predicted Y and uncertianity associated with it. (result of using Bayesian Approach) */

            y = phi_testX.matMul(w_mean);
            // uncertainty in our predictions.
            y_variance = tf.add (tf.scalar(1/beta) , tf.mul(tf.matMul(phi_testX,w_variance),phi_testX).sum(1));

            return {y,y_variance};

        };

        this.evidenceFn = function(x,t,alpha,beta){
            /* compute the evidence function a.k.a log marginal likelihood */
            
            const w_mean      = model.Weights.mean;
            const w_varance   = model.Weights.variance;
            const w_precision = math.inv(w_varance);
            const {0:N , 1:M} = x.shape;
            const E_D = tf.mul( 
                                tf.scalar(beta) , 
                                np.pow( 
                                        np.sum( 
                                                np.sub(
                                                        t ,
                                                        x.matMul(w_mean)
                                                      )
                                               ),
                                        2
                                       )
                              );

            const E_W  = tf.mul(
                                 tf.scalar(beta) ,
                                 tf.sum(
                                        tf.matMul(
                                                w_mean.transpose() ,
                                                w_mean                                                )
                                        )
                                );
                                            
            const E_mN = tf.sum(E_D , E_W);

            const eviFn_p1 = tf.scalar( M*Math.log(alpha) + N*Math.log(beta) - N*Math.log(2*Math.PI)) ;
            const eviFn_p2 = E_mN.neg();
            const eviFn_p3 = tf.log( math.det( w_precision.arraySync() ) ).neg();

            let eviFn =  tf.add( eviFn_p1,eviFn_p2 );
            eviFn     =  tf.add( eviFn,eviFn_p3);
            eviFn     = eviFn.mul(tf.scalar(1/2));

            return eviFn;
        }


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

        this.train = function(){
            // in this block we are going to be calculating the hyperparameters aswell as the 
            // parameter distribution i.e, p(w | t,alpha,beta);

            // TODO: Calculate these using the evidence approximation.
            const alpha = 5e-3;
            const beta  = 11.1;

            const {mean: w_mean,variance: w_variance,precision: w_precision} =  this.paramPDF(cBasisFn(trainX,cBasisFnParams),trainY,alpha,beta);

            /* adding our parameters to our model. */
            model.Weights.mean = w_mean;
            model.Weights.variance = w_variance;
            model.hyperParameters.alpha = alpha;
            model.hyperParameters.beta  = beta;

            return this;
        };

        this.test = function(newtestX = null){

            const phi_testX = cBasisFn( newtestX || testX,cBasisFnParams);
            return this.predictedPDF(phi_testX,model.Weights.mean,model.Weights.variance,model.hyperParameters.beta);
        };

        this.run = function(){
            /* Applying bayesian curve fitting using the given basis fns and tarining data. */

            // TODO: In training we use evidence approx. to calculate our alpha and betas.

            //TODO: make sure that we have all of our data.

            /* hyperParameters:- */

            // ? are they specificially applicable for polynomial kernels? i think not.
            const alpha = 5e-3;
            const beta = 11.1;


            const phiX = cBasisFn(trainX,cBasisFnParams).transpose();

            console.log(phiX.shape)
            const w1 = tf.mul(tf.scalar(alpha), tf.eye(phiX.shape[0])); // uncertainty in parameter
            const w2 = tf.mul(tf.scalar(beta), tf.matMul(phiX, phiX.transpose())); // uncertainty in prediction/data
            const S_inv = tf.add(w1, w2); // eqn 1.72 
            const phi_testX = cBasisFn(testX,cBasisFnParams).transpose();
            
            /* initializing mean and variance */
            const varVec  = tf.zeros(testX.shape).flatten().arraySync();
            const meanVec = varVec.slice(0);

            /* calculating mean and stdev for each test point from our predictive distribution */
            for (let k = 0; k < testX.shape[0]; k++) {
                const phi_testX_k = phi_testX.transpose().slice(k, 1); // taking each test sample one by one. //dim:- [1,8]
                const a  = tf.mul(tf.scalar(beta), phi_testX_k); // dim [1,8]
                const b1 = tf.matMul(phiX, trainY).flatten().arraySync();
                const b2 = S_inv.arraySync();
                const b  = tf.tensor(gauss(b2, b1)).expandDims(1); // solving system of linear equation using gaussian-elemination.

                meanVec[k] = tf.matMul(a, b).flatten().arraySync()[0]; // eqn 1.70

                const c1 = tf.tensor(gauss(S_inv.arraySync(), phi_testX_k.flatten().arraySync())).expandDims(1);
                const c  = tf.matMul(phi_testX_k, c1);

                varVec[k] = tf.add(tf.scalar(1 / beta), c).flatten().arraySync()[0]; // eqn 1.71
            }

            return {meanVec: tf.tensor(meanVec),varVec: tf.tensor(varVec)}

        };

    }
    bcf.prototype = {
        /* parant function:- */

        // Maybe we can use the prototype of a potential Bayes.js Library.
    }

    /* assigning our function to our global Window Object */
    global.BCF = BCF;

    console.log("added BCF");

    }(window)
)







