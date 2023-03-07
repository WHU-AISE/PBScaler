

package org.myproject.ms.monitoring.lgger;

import org.slf4j.MDC;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;


@Configuration
@ConditionalOnProperty(value="spring.sleuth.enabled", matchIfMissing=true)
public class LogAtcfg {

	@Configuration
	@ConditionalOnClass(MDC.class)
	@EnableConfigurationProperties(Slf4jProps.class)
	protected static class Slf4jConfiguration {

		@Bean
		@ConditionalOnProperty(value = "spring.sleuth.log.slf4j.enabled", matchIfMissing = true)
		@ConditionalOnMissingBean
		public ItemLogger slf4jSpanLogger(Slf4jProps sleuthSlf4jProperties) {
			// Sets up MDC entries X-B3-TraceId and X-B3-SpanId
			return new Slf4jItemLogger(sleuthSlf4jProperties.getNameSkipPattern());
		}

		@Bean
		@ConditionalOnProperty(value = "spring.sleuth.log.slf4j.enabled", havingValue = "false")
		@ConditionalOnMissingBean
		public ItemLogger noOpSlf4jSpanLogger() {
			return new NoItemLogger();
		}
	}

	@Bean
	@ConditionalOnMissingClass("org.slf4j.MDC")
	@ConditionalOnMissingBean
	public ItemLogger defaultLoggedSpansHandler() {
		return new NoItemLogger();
	}
}
