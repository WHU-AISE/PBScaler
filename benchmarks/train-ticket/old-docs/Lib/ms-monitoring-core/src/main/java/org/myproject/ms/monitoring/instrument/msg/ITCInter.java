

package org.myproject.ms.monitoring.instrument.msg;

import org.myproject.ms.monitoring.ChainKeys;
import org.myproject.ms.monitoring.Chainer;
import org.springframework.integration.channel.ChannelInterceptorAware;
import org.springframework.integration.channel.interceptor.VetoCapableInterceptor;
import org.springframework.messaging.support.ChannelInterceptor;


class ITCInter extends TCInter implements VetoCapableInterceptor {


	public ITCInter(Chainer tracer, ChainKeys traceKeys,
			MSTMExtra spanExtractor,
			MSTMInject spanInjector) {
		super(tracer, traceKeys, spanExtractor, spanInjector);
	}

	@Override
	public boolean shouldIntercept(String beanName, ChannelInterceptorAware channel) {
		for (ChannelInterceptor interceptor : channel.getChannelInterceptors()) {
			if (interceptor instanceof ATCInter) {
				return false;
			}
		}
		return true;
	}

}
