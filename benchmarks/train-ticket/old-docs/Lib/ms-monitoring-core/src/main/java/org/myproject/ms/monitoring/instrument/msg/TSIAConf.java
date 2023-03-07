

package org.myproject.ms.monitoring.instrument.msg;

import java.util.Random;

import org.springframework.boot.autoconfigure.AutoConfigureAfter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.myproject.ms.monitoring.ChainKeys;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.atcfg.TraceAutoConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.integration.config.GlobalChannelInterceptor;


@Configuration
@ConditionalOnClass(GlobalChannelInterceptor.class)
@ConditionalOnBean(Chainer.class)
@AutoConfigureAfter({ TraceAutoConfiguration.class,
		TSMAConf.class })
@ConditionalOnProperty(value = "spring.sleuth.integration.enabled", matchIfMissing = true)
@EnableConfigurationProperties(ChainKeys.class)
public class TSIAConf {

	@Bean
	@GlobalChannelInterceptor(patterns = "${spring.sleuth.integration.patterns:*}")
	public TCInter traceChannelInterceptor(Chainer tracer,
			ChainKeys traceKeys, Random random, MSTMExtra spanExtractor,
			MSTMInject spanInjector) {
		return new ITCInter(tracer, traceKeys, spanExtractor,
				spanInjector);
	}

}
