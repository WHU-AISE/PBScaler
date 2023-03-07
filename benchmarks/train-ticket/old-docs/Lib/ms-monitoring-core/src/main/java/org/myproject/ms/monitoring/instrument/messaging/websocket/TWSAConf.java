package org.myproject.ms.monitoring.instrument.messaging.websocket;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.AutoConfigureAfter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.myproject.ms.monitoring.ChainKeys;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.instrument.msg.MSTMExtra;
import org.myproject.ms.monitoring.instrument.msg.MSTMInject;
import org.myproject.ms.monitoring.instrument.msg.TCInter;
import org.myproject.ms.monitoring.instrument.msg.TSMAConf;
import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.simp.config.ChannelRegistration;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.config.annotation.AbstractWebSocketMessageBrokerConfigurer;
import org.springframework.web.socket.config.annotation.DelegatingWebSocketMessageBrokerConfiguration;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;


@Component
@Configuration
@AutoConfigureAfter(TSMAConf.class)
@ConditionalOnClass(DelegatingWebSocketMessageBrokerConfiguration.class)
@ConditionalOnBean(Chainer.class)
@ConditionalOnProperty(value = "spring.sleuth.integration.websockets.enabled", matchIfMissing = true)
public class TWSAConf
		extends AbstractWebSocketMessageBrokerConfigurer {

	@Autowired
	Chainer tracer;
	@Autowired
	ChainKeys traceKeys;
	@Autowired
	MSTMExtra spanExtractor;
	@Autowired
	MSTMInject spanInjector;

	@Override
	public void registerStompEndpoints(StompEndpointRegistry registry) {
		// The user must register their own endpoints
	}

	@Override
	public void configureClientOutboundChannel(ChannelRegistration registration) {
		registration.setInterceptors(new TCInter(this.tracer,
				this.traceKeys, this.spanExtractor, this.spanInjector));
	}

	@Override
	public void configureClientInboundChannel(ChannelRegistration registration) {
		registration.setInterceptors(new TCInter(this.tracer,
				this.traceKeys, this.spanExtractor, this.spanInjector));
	}
}