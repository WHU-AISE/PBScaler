

package org.myproject.ms.monitoring.instrument.msg;

import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.myproject.ms.monitoring.ChainKeys;
import org.myproject.ms.monitoring.Chainer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.Message;


@Configuration
@ConditionalOnClass(Message.class)
@ConditionalOnBean(Chainer.class)
public class TSMAConf {

	@Bean
	@ConditionalOnMissingBean
	public MSTMExtra messagingSpanExtractor() {
		return new HBMExtra();
	}

	@Bean
	@ConditionalOnMissingBean
	public MSTMInject messagingSpanInjector(ChainKeys traceKeys) {
		return new HBMInject(traceKeys);
	}
}
