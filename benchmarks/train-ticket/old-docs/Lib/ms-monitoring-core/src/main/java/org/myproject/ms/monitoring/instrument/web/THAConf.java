
package org.myproject.ms.monitoring.instrument.web;

import java.util.regex.Pattern;

import org.springframework.boot.autoconfigure.AutoConfigureAfter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.myproject.ms.monitoring.ChainKeys;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.atcfg.TraceAutoConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;


@Configuration
@ConditionalOnBean(Chainer.class)
@AutoConfigureAfter(TraceAutoConfiguration.class)
@EnableConfigurationProperties({ ChainKeys.class, SWProp.class })
public class THAConf {

	@Bean
	@ConditionalOnMissingBean
	public HTKInject httpTraceKeysInjector(Chainer tracer, ChainKeys traceKeys) {
		return new HTKInject(tracer, traceKeys);
	}

	@Bean
	@ConditionalOnMissingBean
	public HSExtra httpSpanExtractor(SWProp sleuthWebProperties) {
		return new ZHSExtra(Pattern.compile(sleuthWebProperties.getSkipPattern()));
	}

	@Bean
	@ConditionalOnMissingBean
	public HSInject httpSpanInjector() {
		return new ZHSInject();
	}
}
