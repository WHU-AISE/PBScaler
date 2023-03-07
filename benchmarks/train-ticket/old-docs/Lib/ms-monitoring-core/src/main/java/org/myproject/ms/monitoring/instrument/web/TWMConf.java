

package org.myproject.ms.monitoring.instrument.web;

import org.springframework.beans.factory.BeanFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurerAdapter;


@Configuration
class TWMConf extends WebMvcConfigurerAdapter {
	@Autowired BeanFactory beanFactory;

	@Bean
	public THInter traceHandlerInterceptor(BeanFactory beanFactory) {
		return new THInter(beanFactory);
	}

	@Override
	public void addInterceptors(InterceptorRegistry registry) {
		registry.addInterceptor(this.beanFactory.getBean(THInter.class));
	}
}
