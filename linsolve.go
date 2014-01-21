package train

import (
	"math"

	"github.com/reggo/common"
	"github.com/reggo/loss"
	"github.com/reggo/regularize"
	"github.com/reggo/train/diagonal"

	"github.com/gonum/matrix/mat64"
)

// TODO: Still would be nice to have a train.Train method which does the smart stuff

const (
	minGrain = 1
	maxGrain = 500
)

type LossDeriver interface {
	// Gets the current parameters
	//Parameters() []float64

	// Sets the current parameters
	//SetParameters([]float64)

	// Features is either the input or the output from Featurize
	// Deriv will be called after predict so memory may be cached
	Predict(featurizedInput, predOutput []float64)

	// Deriv computes the derivative of the loss with respect
	// to the weight given the predicted output and the derivative
	// of the loss function with respect to the prediction
	Deriv(featurizedInput, predOutput, dLossDPred, dLossDWeight []float64)
}

type Featurizer interface {
	// Featurize transforms the input into the elements of the feature matrix. Feature
	// will have length NumFeatures(). Should not modify input
	Featurize(input, feature []float64)
}

// Linear is a type whose parameters are a linear combination of a set of features
type Trainable interface {
	// NumFeatures returns the number of features
	NumFeatures() int // NumFeatures is how many features the input is transformed into
	InputDim() int
	OutputDim() int
	NumParameters() int        // returns the number of parameters
	Parameters([]float64)      // Puts in place all the parameters into the input
	SetParameters([]float64)   // Sets the new parameters
	NewFeaturizer() Featurizer // Returns a type whose featurize method can be called concurrently
	NewLossDeriver() LossDeriver
	GrainSize() int // Returns the suggested grain size
}

// Creates the features from the inputs. Features must be nSamples x nFeatures or nil
func FeaturizeTrainable(t Trainable, inputs mat64.Matrix, features *mat64.Dense) *mat64.Dense {
	nSamples, nDim := inputs.Dims()
	if features == nil {
		nFeatures := t.NumFeatures()
		features = mat64.NewDense(nSamples, nFeatures, nil)
	}

	rowViewer, isRowViewer := inputs.(mat64.RowViewer)
	var f func(start, end int)
	if isRowViewer {
		f = func(start, end int) {
			featurizer := t.NewFeaturizer()
			for i := start; i < end; i++ {
				featurizer.Featurize(rowViewer.RowView(i), features.RowView(i))
			}
		}
	} else {
		f = func(start, end int) {
			featurizer := t.NewFeaturizer()
			input := make([]float64, nDim)
			for i := start; i < end; i++ {
				for j := range input {
					input[i] = inputs.At(i, j)
				}
				featurizer.Featurize(input, features.RowView(i))
			}
		}
	}

	common.ParallelFor(nSamples, common.GetGrainSize(nSamples, minGrain, maxGrain), f)
	return features
}

// IsLinearRegularizer returns true if the regularizer can be used with LinearSolve
func IsLinearSolveRegularizer(r regularize.Regularizer) bool {
	switch r.(type) {
	case nil:
	case regularize.None:
	default:
		return false
	}
	return true
}

func IsLinearSolveLosser(l loss.Losser) bool {
	switch l.(type) {
	case nil:
	case loss.SquaredDistance:
	default:
		return false
	}
	return true
}

type MulMatrix interface {
	mat64.Muler
	mat64.Matrix
}

// LinearSolve trains a Linear algorithm.
// Assumes inputs and outputs are already scaled
// If features is nil will call featurize
// Will return nil if regularizer is not a linear regularizer
// Is destructive if any of the weights are zero
func LinearSolve(l Trainable, features *mat64.Dense, inputs, trueOutputs mat64.Matrix, weights []float64, r regularize.Regularizer) (parameters *mat64.Dense) {
	// TODO: Allow tikhonov regularization
	// TODO: Add test for weights

	if !IsLinearSolveRegularizer(r) {
		return nil
	}

	if features == nil {
		features = FeaturizeTrainable(l, inputs, features)
	}

	_, nFeatures := features.Dims()

	var weightedFeatures, weightedOutput *mat64.Dense

	if weights != nil {
		scaledWeight := make([]float64, len(weights))
		for i, weight := range weights {
			scaledWeight[i] = math.Sqrt(weight)
		}

		diagWeight := diagonal.NewDiagonal(nFeatures, weights)

		nSamples, outputDim := trueOutputs.Dims()
		weightedOutput = mat64.NewDense(nSamples, outputDim, nil)
		weightedFeatures = mat64.NewDense(nSamples, nFeatures, nil)

		weightedOutput.Mul(diagWeight, trueOutputs)
		weightedFeatures.Mul(diagWeight, features)
	}

	switch r.(type) {
	case nil:
	case regularize.None:
	default:
		panic("Shouldn't be here. Must be error in IsLinearRegularizer")
	}

	if weights == nil {
		parameters = mat64.Solve(features, trueOutputs)
		return parameters
	}
	parameters = mat64.Solve(weightedFeatures, weightedOutput)
	return parameters
}
