package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.layers.recurrent.LSTMHelpers;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class RecurrentBatchNormalization extends AbstractLSTM {

    private double forgetGateBiasInit;
    private IActivation gateActivationFn = new ActivationSigmoid();

    private RecurrentBatchNormalization(RecurrentBatchNormalization.Builder builder) {
        super(builder);
        this.forgetGateBiasInit = builder.forgetGateBiasInit;
        this.gateActivationFn = builder.gateActivationFn;

        initializeConstraints(builder);
    }

    @Override
    protected void initializeConstraints(org.deeplearning4j.nn.conf.layers.Layer.Builder<?> builder){
        super.initializeConstraints(builder);
        if(((RecurrentBatchNormalization.Builder)builder).recurrentConstraints != null){
            if(constraints == null){
                constraints = new ArrayList<>();
            }
            for (LayerConstraint c : ((RecurrentBatchNormalization.Builder) builder).recurrentConstraints) {
                LayerConstraint c2 = c.clone();
                c2.setParams(Collections.singleton(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY));
                constraints.add(c2);
            }
        }
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("RecurrentBatchNormalization", getLayerName(), layerIndex, getNIn(), getNOut());
        org.deeplearning4j.nn.layers.recurrent.RecurrentBatchNormalization ret =
                new org.deeplearning4j.nn.layers.recurrent.RecurrentBatchNormalization(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return GravesLSTMParamInitializer.getInstance();
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //TODO - CuDNN etc
        return LSTMHelpers.getMemoryReport(this, inputType);
    }

    @AllArgsConstructor
    public static class Builder extends AbstractLSTM.Builder<RecurrentBatchNormalization.Builder> {

        @SuppressWarnings("unchecked")
        public RecurrentBatchNormalization build() {
            return new RecurrentBatchNormalization(this);
        }
    }

}