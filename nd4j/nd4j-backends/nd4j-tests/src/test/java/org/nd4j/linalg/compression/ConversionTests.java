/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.compression;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

@Slf4j
@RunWith(Parameterized.class)
public class ConversionTests extends BaseNd4jTest {

    public ConversionTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    DataBuffer.Type initialType;

    @After
    public void after() {
        Nd4j.setDataType(this.initialType);
    }


    @Test
    public void testDoubleToFloats1() {
        val dtype = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        val arrayX = Nd4j.create(10).assign(1.0);


        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        val arrayY = Nd4j.create(10).assign(1.0);


        val converted = arrayX.convertToFloats();
        val exp = Nd4j.create(10).assign(2.0);
        converted.addi(arrayY);

        assertEquals(exp, converted);



        Nd4j.setDataType(dtype);
    }


    @Test
    public void testFloatsToDoubles1() {
        val dtype = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        val arrayX = Nd4j.create(10).assign(1.0);


        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        val arrayY = Nd4j.create(10).assign(1.0);


        val converted = arrayX.convertToDoubles();
        val exp = Nd4j.create(10).assign(2.0);
        converted.addi(arrayY);

        assertEquals(exp, converted);



        Nd4j.setDataType(dtype);
    }

    @Test
    public void testFloatsToHalfs1() {
        if (Nd4j.getExecutioner().type() != OpExecutioner.ExecutionerType.CUDA)
            return;

        val dtype = Nd4j.dataType();

        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        val arrayX = Nd4j.create(10).assign(1.0);


        Nd4j.setDataType(DataBuffer.Type.HALF);
        val arrayY = Nd4j.create(10).assign(1.0);


        val converted = arrayX.convertToHalfs();
        val exp = Nd4j.create(10).assign(2.0);
        converted.addi(arrayY);

        assertEquals(exp, converted);



        Nd4j.setDataType(dtype);
    }

    @Test
    public void testConvDouble_1(){
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        val exp = Nd4j.linspace(-5, 5, 11);

        DataBuffer.Type[] types = null;
        if (Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA) {
            types = new DataBuffer.Type[]{DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT, DataBuffer.Type.HALF};
        } else {
            types = new DataBuffer.Type[]{DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT};
        }

        for(val t : types) {
            Nd4j.setDataType(t);

            val arr = Nd4j.linspace(-5, 5, 11);
            val arr2 = arr.convertToDoubles();

            Nd4j.setDataType(DataBuffer.Type.DOUBLE);
            assertEquals("Fail with dtype [" + t + "]", exp, arr2);
        }
    }

    @Test
    public void testConvFloat_1(){
        Nd4j.setDataType(DataBuffer.Type.FLOAT);
        val exp = Nd4j.linspace(-5, 5, 11);

        DataBuffer.Type[] types = null;
        if (Nd4j.getExecutioner().type() == OpExecutioner.ExecutionerType.CUDA) {
            types = new DataBuffer.Type[]{DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT, DataBuffer.Type.HALF};
        } else {
            types = new DataBuffer.Type[]{DataBuffer.Type.DOUBLE, DataBuffer.Type.FLOAT};
        }

        for(val t : types) {
            Nd4j.setDataType(t);

            val arr = Nd4j.linspace(-5, 5, 11);
            val arr2 = arr.convertToFloats();

            Nd4j.setDataType(DataBuffer.Type.FLOAT);
            assertEquals("Fail with dtype [" + t + "]", exp, arr2);
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
