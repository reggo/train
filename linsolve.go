package train

import (
	"math"

	"github.com/reggo/common"
	"github.com/reggo/loss"
	"github.com/reggo/regularize"

	"github.com/gonum/matrix/mat64"

	"mattest"
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
	case regularize.None:
		return true
	default:
		return false
	}
}

func IsLinearSolveLosser(l loss.Losser) bool {
	switch l.(type) {
	case loss.SquaredDistance:
		return true
	default:
		return false
	}
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
func LinearSolve(l Linear, features *mat64.Dense, inputs, trueOutputs MulMatrix, weights []float64, r regularize.Regularizer) (parameters *mat64.Dense) {
	// TODO: Allow tikhonov regularization
	// TODO: Add test for weights

	if !IsLinearSolveRegularizer(r) {
		return nil
	}

	if features == nil {
		features = FeaturizeLinear(l, inputs, features)
	}

	_, nFeatures := features.Dims()

	if weights != nil {
		for i, weight := range weights {
			weights[i] = math.Sqrt(weight)
		}
		diagWeight := mattest.NewDiagonal(nFeatures, weights)

		trueOutputs.Mul(diagWeight, trueOutputs)
		features.Mul(diagWeight, features)

		defer func() {
			// Unscale by the weights
			err := diagWeight.Inv(diagWeight)
			if err != nil {
				panic(err)
			}
			// TODO: Figure out what to do with zero weights
			trueOutputs.Mul(diagWeight, trueOutputs)
			features.Mul(diagWeight, features)
			for i, weight := range weights {
				weights[i] = weight * weight

			}
		}()

	}

	switch r.(type) {
	case regularize.None:
	default:
		panic("Shouldn't be here. Must be error in IsLinearRegularizer")
	}

	parameters = mat64.Solve(features, trueOutputs)

	return parameters
}
