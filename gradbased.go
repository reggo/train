package train

import (
	"sync"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"

	"github.com/reggo/common"
	"github.com/reggo/loss"
	"github.com/reggo/regularize"
)

type lossDerivStruct struct {
	loss  float64
	deriv []float64
}

// BatchGradBased is a wrapper for training a trainable with
// a fixed set of samples
type BatchGradBased struct {
	t           Trainable
	losser      loss.DerivLosser
	regularizer regularize.Regularizer

	inputDim    int
	outputDim   int
	nTrain      int
	nParameters int
	grainSize   int

	inputs  mat64.Matrix
	outputs mat64.Matrix

	features *mat64.Dense

	lossDerivFunc func(start, end int, c chan lossDerivStruct)
}

// NewBatchGradBased creates a new batch grad based with the given inputs
func NewBatchGradBased(t Trainable, precompute bool, inputs, outputs mat64.Matrix, losser loss.DerivLosser, regularizer regularize.Regularizer) *BatchGradBased {
	var features *mat64.Dense
	if precompute {
		FeaturizeTrainable(t, inputs, features)
	}

	// TODO: Add in error checking

	nTrain, outputDim := outputs.Dims()
	_, inputDim := inputs.Dims()
	g := &BatchGradBased{
		t:           t,
		inputs:      inputs,
		outputs:     outputs,
		features:    features,
		losser:      losser,
		regularizer: regularizer,
		nTrain:      nTrain,
		outputDim:   outputDim,
		inputDim:    inputDim,
		nParameters: t.NumParameters(),
		grainSize:   t.GrainSize(),
	}

	// TODO: Add in row viewer stuff
	// TODO: Create a different function for computing just the loss
	//inputRowViewer, ok := inputs.(mat64.RowViewer)
	//outputRowViewer, ok := outputs.(mat64.RowViewer)

	// TODO: Move this to its own function
	var f func(start, end int, c chan lossDerivStruct)

	switch {
	default:
		panic("Shouldn't be here")
	case precompute:
		f = func(start, end int, c chan lossDerivStruct) {
			lossDeriver := g.t.NewLossDeriver()
			prediction := make([]float64, g.outputDim)
			dLossDPred := make([]float64, g.outputDim)
			dLossDWeight := make([]float64, g.nParameters)
			totalDLossDWeight := make([]float64, g.nParameters)
			var loss float64
			output := make([]float64, g.outputDim)
			for i := start; i < end; i++ {
				// Compute the prediction
				lossDeriver.Predict(g.features.RowView(i), prediction)
				// Compute the loss
				for j := range output {
					output[j] = g.outputs.At(i, j)
				}
				loss += g.losser.LossDeriv(prediction, output, dLossDPred)
				// Compute the derivative
				lossDeriver.Deriv(g.features.RowView(i), prediction, dLossDPred, dLossDWeight)

				floats.Add(totalDLossDWeight, dLossDWeight)

				// Send the value back on the channel
				c <- lossDerivStruct{
					loss:  loss,
					deriv: totalDLossDWeight,
				}
			}
		}
	case !precompute:
		f = func(start, end int, c chan lossDerivStruct) {
			lossDeriver := g.t.NewLossDeriver()
			prediction := make([]float64, g.outputDim)
			dLossDPred := make([]float64, g.outputDim)
			dLossDWeight := make([]float64, g.nParameters)
			totalDLossDWeight := make([]float64, g.nParameters)
			var loss float64
			output := make([]float64, g.outputDim)

			input := make([]float64, g.inputDim)

			features := make([]float64, g.t.NumFeatures())

			featurizer := g.t.NewFeaturizer()
			for i := start; i < end; i++ {
				// featurize the input
				for j := range input {
					input[j] = g.inputs.At(i, j)
				}

				featurizer.Featurize(input, features)

				// Compute the prediction
				lossDeriver.Predict(features, prediction)
				// Compute the loss
				for j := range output {
					output[j] = g.outputs.At(i, j)
				}
				loss += g.losser.LossDeriv(prediction, output, dLossDPred)

				// Compute the derivative
				lossDeriver.Deriv(features, prediction, dLossDPred, dLossDWeight)

				// Add to the total derivative
				floats.Add(totalDLossDWeight, dLossDWeight)

				// Send the value back on the channel
				c <- lossDerivStruct{
					loss:  loss,
					deriv: totalDLossDWeight,
				}
			}
		}
	}

	g.lossDerivFunc = f

	return g
}

// ObjDeriv computes the objective value and stores the derivative in place
func (g *BatchGradBased) ObjDeriv(parameters []float64, derivative []float64) (loss float64) {
	c := make(chan lossDerivStruct, 10)

	// Set the channel for parallel for
	f := func(start, end int) {
		g.lossDerivFunc(start, end, c)
	}

	go func() {
		wg := &sync.WaitGroup{}
		// Compute the losses and the derivatives all in parallel
		wg.Add(2)
		go func() {
			common.ParallelFor(g.nTrain, g.grainSize, f)
			wg.Done()
		}()
		// Compute the regularization
		go func() {
			deriv := make([]float64, g.nParameters)
			loss := g.regularizer.LossDeriv(parameters, deriv)
			c <- lossDerivStruct{
				loss:  loss,
				deriv: deriv,
			}
			wg.Done()
		}()
		// Wait for all of the results to be sent on the channel
		wg.Wait()
		// Close the channel
		close(c)
	}()
	// zero the derivative
	for i := range derivative {
		derivative[i] = 0
	}

	// Range over the channel, incrementing the loss and derivative
	// as they come in
	for l := range c {
		loss += l.loss
		floats.Add(derivative, l.deriv)
	}
	// Normalize by the number of training samples
	loss /= float64(g.nTrain)
	floats.Scale(float64(g.nTrain), derivative)
	return loss
}
