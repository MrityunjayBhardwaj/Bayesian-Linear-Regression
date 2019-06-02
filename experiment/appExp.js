const nSamples = 15;
const ntestSamples = 10;
const maxDegree = 10;  // maximum degree of polynomial we can use.

/* Generating Synthetic Data:- */
const trainX = tf.linspace(0.1,1.0, nSamples).expandDims(1); // returns a tf.tensor
const noisefactor = 0.0;
const trainY = tf.add(tf.sin(trainX.mul(tf.scalar(2 * Math.PI))), tf.mul(tf.scalar(noisefactor), tf.truncatedNormal(trainX.shape))); //sin(2*pi*x) + e (some noise) bcuz its an emperical distribution. 
const testX = tf.linspace(-0.5, 1.0, ntestSamples).expandDims(1);
const testY = tf.sin(testX.mul(tf.scalar(2 * Math.PI))); //sin(2*pi*x)


const bcf = LBF(trainX,trainY,testX,testY);
// const {beta: newBeta,alpha: newAlpha} = bcf.evidenceMaximization(trainX,trainY,0.0001,0.0);
bcf.useBasisFn("polynomial",{degree: 8});

// let alpha =  5e-3;
// let beta  =  11.1111;

// console.log(newBeta,newAlpha);
let { y : meanVec, yVariance : varVec} = bcf.train().test();

// const genYsamples = bcf.genY(5);

meanVec.print();
varVec.print();

// const beta = 1 / (0.3 ** 2)
// const alpha = 0.005
// const predictedY = LBF(trainX,trainY,testX,testY).train().test();
// const predictedY = LBF(trainX,trainY,testX,testY).useBasisFn("polynomial").train().test();

// experimental :-
// for(let i=0;i<1;i++){
//     i = 4; 
//     bcf.useBasisFn("polynomial",{degree: i}).train();
//     let k = bcf.evidenceFn(trainX,trainY,alpha,beta);
//     // k.print();
// }

// const lbf = LBF(trainX,trainY,testX,testY).learnBasisFn().train().test();
