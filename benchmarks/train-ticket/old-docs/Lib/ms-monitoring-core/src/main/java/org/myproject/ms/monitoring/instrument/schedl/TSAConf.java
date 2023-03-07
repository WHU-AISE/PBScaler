

package org.myproject.ms.monitoring.instrument.schedl;

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
import org.springframework.context.annotation.EnableAspectJAutoProxy;

import java.util.regex.Pattern;


@Configuration
@EnableAspectJAutoProxy
@ConditionalOnProperty(value = "spring.sleuth.scheduled.enabled", matchIfMissing = true)
@ConditionalOnBean(Chainer.class)
@AutoConfigureAfter(TraceAutoConfiguration.class)
@EnableConfigurationProperties(SSProp.class)
public class TSAConf {

	@ConditionalOnClass(name = "org.aspectj.lang.ProceedingJoinPoint")
	@Bean
	public TSAspect traceSchedulingAspect(Chainer tracer, ChainKeys traceKeys,
			SSProp sleuthSchedulingProperties) {
		return new TSAspect(tracer, traceKeys, Pattern.compile(sleuthSchedulingProperties.getSkipPattern()));
	}
}
