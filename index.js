function neuralNetwork() {

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
        return zeros(nInputs).map(() => Math.random() * 2.0 - 1.0);
    }

    function neuronWeight(neuron, index) {
        return neuron[index];
    }

    function neuronForward(neuron, inputs) {
        return activation(inputs.reduce((sum, input, index) => {
            return sum + input * neuronWeight(neuron, index);
        }, 0));
    }

    function layerForward(layer, inputs) {
        return layer.map((neuron) => neuronForward(neuron, inputs));
    }

    function layerForwardArray(layer, inputsArray) {
        return inputsArray.map(layerForward.bind(null, layer));
    }

    function neuronForwardArray(neuron, inputsArray) {
        return inputsArray.map(neuronForward.bind(null, neuron));
    }

    function createLayer(nInputs, nNeurons) {
        return zeros(nNeurons).map(() => createNeuron(nInputs));
    }

    function layerNeuron(layer, index) {
        return layer[index];
    }

    function create(nInputs, nNeurons) {
        if (nNeurons > 1)
            return [
                createLayer(nInputs, nNeurons),
                createLayer(nNeurons, 1)
            ];
        else
            return [ createLayer(nInputs, 1) ]
    }

    function train(nIteration, network, trainingInputsArray, trainingOutputArray) {
        while (nIteration-- > 0)
            network = trainIteration(network, trainingInputsArray, trainingOutputArray);
        return network;
    }

    function forward(network, inputs) {
        return network.reduce(
            (x, layer) => layerForward(layer, x),
            inputs);
    }

    function forwardArray(network, inputsArray) {
        return inputsArray.map(forward.bind(null, network));
    }

    function test(name, network, testInputsArray, testOutputsArray) {
        const actual = forwardArray(network, testInputsArray).map(Math.round);
        const expect = transpose(testOutputsArray)[0];
        const stats = actual.reduce((stats, actual, index) => {
            const ok = (Math.round(actual) == expect[index] ? 1 : 0);
            const success = stats.success + ok;
            const failure = stats.failure + 1 - ok;
            const total = stats.total + 1;
            return { success, failure, total };
        }, { success:0, failure:0, total:0 });
        //console.log("actual: ", );
        //console.log("expect: ", transpose(testOutputsArray)[0]);
        console.log(" === " + name + " === ");
        console.log("Success rate: " + Math.round(1000 * stats.success / stats.total) / 10 + " %");
        console.log("  Error rate: " + Math.round(1000 * stats.failure / stats.total) / 10 + " %");
    }

    function dot(m1 /* h1 x w1 */, m2 /* h2 x w2 */) {
        if (m1[0].length != m2.length) {
            throw new Error("dot impossible: w1 != h2");
        }
        // for each row,
        return m1.map((row1, rowIndex) => {
            // produce a new row of size w2
            return m2[0].map((dummy, colIndex) => {
                return m2.reduce((sum, row2, rowIndex2) => {
                    return sum + row1[rowIndex2] * row2[colIndex];
                }, 0);
            });
        });
    }

    function mul(coef, m) {
        if (typeof coef == 'number')
            return m.map((row) => row.map((cell) => cell * coef));
        else {
            if (m.width != coef.width || m[0].width != coef[0].width)
                throw new Error("cant mul matrices of different sizes");
            return m.map((row, rowIndex) => row.map((cell, colIndex) => cell * coef[rowIndex][colIndex]));
        }
    }

    function sub(m1, m2) {
        if (m1.width != m2.width || m1[0].width != m2[0].width)
            throw new Error("cant substract matrices of different sizes");
        return m1.map((row, rowIndex) => row.map((cell, colIndex) => cell - m2[rowIndex][colIndex]));
    }

    function add(m1, m2) {
        if (m1.width != m2.width || m1[0].width != m2[0].width)
            throw new Error("cant add matrices of different sizes");
        return m1.map((row, rowIndex) => row.map((cell, colIndex) => cell + m2[rowIndex][colIndex]));
    }

    function applyAll(fn, m) {
        return m.map((row) => row.map((cell) => fn(cell)));
    }

    function transpose(m) {
        return m[0].map((dummy, colIndex) => m.reduce((out, row, rowIndex) => out.concat(row[colIndex]), []));
    }

    function debugMatrix(matrix, name, fullDisplay) {
        if (typeof matrix[0][0] != 'number') {
            console.log(name + " IS NOT A MATRIX!");
            console.log(matrix);
            return;
        }
        console.log(name + " (" + matrix.length + "x" + matrix[0].length + ")");
        const hasNaN = matrix.reduce((hasNaN, row) => {
            return row.reduce((hasNaN, cell) => {
                return hasNaN || isNaN(cell) || (cell == undefined);
            }, hasNaN);
        }, false);
        if (hasNaN) {
            console.log(name + " IS BUGGY!");
            fullDisplay = true;
        }
        if (fullDisplay)
            console.log(matrix);
    }

    function trainIteration(network, trainingInputs, trainingOutput) {

        if (network.length == 1) {
            return [ [ neuronTrainIteration(network[0][0], trainingInputs, transpose(trainingOutput)[0]) ] ];
        }
        else if (network.length == 2) {
            let x = trainingInputs;
            const outputs = network.map((layer) => x = layerForwardArray(layer, x));
            const inputs = [ trainingInputs ].concat(outputs.slice(0, outputs.length - 1));

            //debugMatrix(trainingInputs, "trainingInputs");
            //debugMatrix(trainingOutput, "trainingOutput", true);
            //debugMatrix(network[0], "network[0]");
            //debugMatrix(network[1], "network[1]");
            //debugMatrix(inputs[0],  "inputs[0]");
            //debugMatrix(inputs[1],  "inputs[1]");
            //debugMatrix(outputs[0], "outputs[0]");
            //debugMatrix(outputs[1], "outputs[1]", true);

            const error = [];
            const delta = [];

            error[1] = sub(trainingOutput, outputs[1]);
            delta[1] = mul(error[1], applyAll(activationDerivative, outputs[1]));

            error[0] = dot(delta[1], network[1]);
            delta[0] = mul(error[0], applyAll(activationDerivative, outputs[0]));

            const weightOffsets = [];

            weightOffsets[1] = transpose(dot(transpose(inputs[1]), delta[1]));
            weightOffsets[0] = transpose(dot(transpose(inputs[0]), delta[0]));

            //debugMatrix(error[1], "error[1]", true);
            //debugMatrix(delta[1], "delta[1]");
            //debugMatrix(error[0], "error[0]");
            //debugMatrix(delta[0], "delta[0]");
            //debugMatrix(weightOffsets[1], "weightOffsets[1]");
            //debugMatrix(weightOffsets[0], "weightOffsets[0]");

            return [
                add(network[0], weightOffsets[0]),
                add(network[1], weightOffsets[1])
            ];
        }
        else {
            return network;
        }
    }

    function neuronTrainIteration(neuron, trainingInputsArray, trainingOutputArray) {
        const outputArray /* 1xN */ = neuronForwardArray(neuron, trainingInputsArray);
        const errorArray /* 1xN */ = outputArray.map((output, index) => trainingOutputArray[index] - output);
        const deltaArray /* 1xN */ = outputArray.map((output, index) => activationDerivative(output) * errorArray[index]);
        // weightOffsets = transposed(trainingInputsArray) . deltaArray
        const weightOffsets /* 1xM */ = neuron.map((weight, weightIndex) => {
            return deltaArray.reduce((sum, delta, deltaIndex) => {
                return sum + delta * trainingInputsArray[deltaIndex][weightIndex];
            }, 0);
        });
        return neuron.map((weight, index) => weight + weightOffsets[index]);
    }

    return {
        create, test, train
    };
}

const { create, test, train } = neuralNetwork();

function problem1() {
    // most basic: copy over value from first column
    const testInputsArray  = [[1,0,0],[0,1,1],[1,1,1]];
    const testOutputsArray = [ [1], [0], [1] ];

    const trainingInputsArray = [[ 1, 0, 0 ], [ 0, 1, 1 ], [ 1, 1, 1], [ 0, 0, 1], [ 1, 1, 0 ], [ 0, 1, 0 ]];
    const trainingOutputArray = [ [1], [0], [1], [0], [1], [0] ];

    const network = create(3, 1);
    const trainedNetwork = train(500, network, trainingInputsArray, trainingOutputArray);

    test("problem1", trainedNetwork, testInputsArray, testOutputsArray);
}

function problem2() {
    // 1bit sum of the 3 columns
    const testInputsArray = [[1,0,0],[0,1,1],[1,1,1],[1,1,0]];
    const testOutputsArray = [ [1], [0], [1], [0] ];

    const trainingInputsArray = [[ 1, 0, 0 ], [ 0, 1, 1 ], [ 1, 1, 1], [ 0, 0, 1], [ 1, 1, 0 ], [ 0, 1, 0 ]];
    const trainingOutputArray = [ [1], [0], [1], [1], [0], [1] ];

    const network = create(3, 4);
    const trainedNetwork = train(500, network, trainingInputsArray, trainingOutputArray);

    test("problem2", trainedNetwork, testInputsArray, testOutputsArray);
}

problem1();
problem2();
