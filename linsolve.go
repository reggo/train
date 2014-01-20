package train

import (
	"math"

	"github.com/reggo/common"
	"github.com/reggo/loss"
	"github.com/reggo/regularize"
	"github.com/reggo/train/diagonal"

	"github.com/gonum/matrix/mat64"
)

const (
	minGrain = 1
	maxGrain = 500
)

// Linear is a type whose parameters are a linear combination of a set of features
type Linear interface {
	// NumFeatures returns the number of features
	NumFeatures() int

	// Featurize transforms the input into the elements of the feature matrix. Feature
	// will have length NumFeatures()
	Featurize(input, feature []float64)

	// CanParallelize returns true if Featurize can be called in parallel
	CanParallelize() bool
}

// Creates the features from the inputs. Features must be nSamples x nFeatures or nil
func FeaturizeLinear(l Linear, inputs mat64.Matrix, features *mat64.Dense) *mat64.Dense {
	nSamples, nDim := inputs.Dims()
	if features == nil {
		nFeatures := l.NumFeatures()
		features = mat64.NewDense(nSamples, nFeatures, nil)
	}

	rowViewer, isRowViewer := inputs.(mat64.RowViewer)
	var f func(start, end int)
	if isRowViewer {
		f = func(start, end int) {
			for i := start; i < end; i++ {
				l.Featurize(rowViewer.RowView(i), features.RowView(i))
			}
		}
	} else {
		f = func(start, end int) {
			input := make([]float64, nDim)
			for i := start; i < end; i++ {
				for j := range input {
					input[i] = inputs.At(i, j)
				}
				l.Featurize(input, features.RowView(i))
				// Don't need to set because Featurize doesn't modify
			}
		}
	}

	if l.CanParallelize() {
		common.ParallelFor(nSamples, common.GetGrainSize(nSamples, minGrain, maxGrain), f)
	} else {
		f(0, nSamples)
	}
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
func LinearSolve(l Linear, features *mat64.Dense, inputs, trueOutputs mat64.Matrix, weights []float64, r regularize.Regularizer) (parameters *mat64.Dense) {
	// TODO: Allow tikhonov regularization
	// TODO: Add test for weights

	if !IsLinearSolveRegularizer(r) {
		return nil
	}

	if features == nil {
		features = FeaturizeLinear(l, inputs, features)
	}

	_, nFeatures := features.Dims()

	var weightedFeatures, weightedOutput *mat64.Dense

	if weights != nil {

		nSamples, outputDim := trueOutputs.Dims()
		weightedOutput = mat64.NewDense(nSamples, outputDim, nil)
		weightedOutput.Copy(trueOutputs)

		weightedFeatures = mat64.NewDense(nSamples, nFeatures, nil)
		weightedFeatures.Copy(features)

		scaledWeight := make([]float64, len(weights))
		for i, weight := range weights {
			scaledWeight[i] = math.Sqrt(weight)
		}

		diagWeight := diagonal.NewDiagonal(nFeatures, weights)

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
