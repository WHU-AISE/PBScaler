

package org.myproject.ms.monitoring.instrument.async;

import org.myproject.ms.monitoring.instrument.schedl.TSAConf;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.BeanFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.boot.autoconfigure.AutoConfigureAfter;
import org.springframework.boot.autoconfigure.AutoConfigureBefore;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.AsyncConfigurer;


@Configuration
@ConditionalOnBean(AsyncConfigurer.class)
@AutoConfigureBefore(ADAtcfg.class)
@ConditionalOnProperty(value = "spring.sleuth.async.enabled", matchIfMissing = true)
@AutoConfigureAfter(TSAConf.class)
public class ACAtcfg implements BeanPostProcessor {

	@Autowired
	private BeanFactory beanFactory;

	@Override
	public Object postProcessBeforeInitialization(Object bean, String beanName)
			throws BeansException {
		return bean;
	}

	@Override
	public Object postProcessAfterInitialization(Object bean, String beanName)
			throws BeansException {
		if (bean instanceof AsyncConfigurer) {
			AsyncConfigurer configurer = (AsyncConfigurer) bean;
			return new LTACus(this.beanFactory, configurer);
		}
		return bean;
	}

}