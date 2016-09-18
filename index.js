function initLib() {

    // An array of n zeros
    function zeros(n) {
        var ret = [];
        while(ret.length < n) ret.push(0);
        return ret;
    }

    function sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    function sigmoidDerivative(x) {
        return x * (1 - x);
    }

    const activation = sigmoid;
    const activationDerivative = sigmoidDerivative;

    // Create a randomly initialized neuron with n synapse (inputs)
    function createNeuron(nInputs) {
        return zeros(nInputs).map((x) => Math.random() * 2.0 - 1.0);
    }

    function inputWeight(neuron, index) {
        return neuron[index];
    }

    function forward(neuron, inputs) {
        return activation(inputs.reduce((sum, input, index) => {
            return sum + input * inputWeight(neuron, index);
        }, 0));
    }

    function forwardArray(neuron, inputsArray) {
        return inputsArray.map(forward.bind(null, neuron));
    }

    function test(neuron, testInputsArray, testOutputsArray) {
        console.log("actual: ", forwardArray(neuron, testInputsArray));
        console.log("expect: ", testOutputsArray);
    }

    function train(nIteration, neuron, trainingInputsArray, trainingOutputArray) {
        while (nIteration-- > 0)
            neuron = trainIteration(neuron, trainingInputsArray, trainingOutputArray);
        return neuron;
    }

    function trainIteration(neuron, trainingInputsArray, trainingOutputArray) {
        const outputArray = forwardArray(neuron, trainingInputsArray);
        const errorArray = outputArray.map((output, index) => trainingOutputArray[index] - output);
        const deltaArray = outputArray.map((output, index) => activationDerivative(output) * errorArray[index]);
        const weightOffsets = neuron.map((weight, weightIndex) => {
            return deltaArray.reduce((sum, delta, deltaIndex) => {
                return sum + delta * trainingInputsArray[deltaIndex][weightIndex];
            }, 0);
        });
        return neuron.map((weight, index) => weight + weightOffsets[index]);
    }

    const create = createNeuron;

    return {
        create, test, train
    };
}

const { create, test, train } = initLib();

const testInputsArray = [[1,0,0],[0,1,1],[1,1,1]];
const testOutputsArray = [ 1, 0, 1 ];

const trainingInputsArray = [[ 1, 0, 0 ], [ 0, 1, 1 ], [ 1, 1, 1], [ 0, 0, 1], [ 1, 1, 0 ], [ 0, 1, 0 ]];
const trainingOutputArray = [ 1, 0, 1, 0, 1, 0 ];

const network = create(3, 1);
const trainedNetwork = train(10000, network, trainingInputsArray, trainingOutputArray);

console.log("Before:", network);
test(network, testInputsArray, testOutputsArray);
console.log("After:", trainedNetwork);
test(trainedNetwork, testInputsArray, testOutputsArray);

