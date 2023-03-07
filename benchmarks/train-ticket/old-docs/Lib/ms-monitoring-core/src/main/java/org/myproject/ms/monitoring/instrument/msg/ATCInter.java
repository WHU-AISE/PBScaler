package org.myproject.ms.monitoring.instrument.msg;

import java.lang.invoke.MethodHandles;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.ItemTextMap;
import org.myproject.ms.monitoring.ChainKeys;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.util.ItemNameUtil;
import org.springframework.integration.channel.AbstractMessageChannel;
import org.springframework.integration.context.IntegrationObjectSupport;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.support.ChannelInterceptorAdapter;
import org.springframework.messaging.support.ExecutorChannelInterceptor;
import org.springframework.util.ClassUtils;


abstract class ATCInter extends ChannelInterceptorAdapter
		implements ExecutorChannelInterceptor {

	private static final Log log = LogFactory.getLog(MethodHandles.lookup().lookupClass());

	
	protected static final String MESSAGE_COMPONENT = "message";

	private final Chainer tracer;
	private final ChainKeys traceKeys;
	private final MSTMExtra spanExtractor;
	private final MSTMInject spanInjector;

	protected ATCInter(Chainer tracer, ChainKeys traceKeys,
			MSTMExtra spanExtractor,
			MSTMInject spanInjector) {
		this.tracer = tracer;
		this.traceKeys = traceKeys;
		this.spanExtractor = spanExtractor;
		this.spanInjector = spanInjector;
	}

	protected Chainer getTracer() {
		return this.tracer;
	}

	protected ChainKeys getTraceKeys() {
		return this.traceKeys;
	}

	protected MSTMInject getSpanInjector() {
		return this.spanInjector;
	}

	
	protected Item buildSpan(ItemTextMap carrier) {
		try {
			return this.spanExtractor.joinTrace(carrier);
		} catch (Exception e) {
			log.error("Exception occurred while trying to extract span from carrier", e);
			return null;
		}
	}

	String getChannelName(MessageChannel channel) {
		String name = null;
		if (ClassUtils.isPresent(
				"org.springframework.integration.context.IntegrationObjectSupport",
				null)) {
			if (channel instanceof IntegrationObjectSupport) {
				name = ((IntegrationObjectSupport) channel).getComponentName();
			}
			if (name == null && channel instanceof AbstractMessageChannel) {
				name = ((AbstractMessageChannel) channel).getFullChannelName();
			}
		}
		if (name == null) {
			name = channel.toString();
		}
		return name;
	}

	String getMessageChannelName(MessageChannel channel) {
		return ItemNameUtil.shorten(MESSAGE_COMPONENT + ":" + getChannelName(channel));
	}

}
