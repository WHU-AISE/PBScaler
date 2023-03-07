

package org.myproject.ms.monitoring.instrument.web;

import javax.servlet.http.HttpServletRequest;
import java.lang.invoke.MethodHandles;
import java.util.Collections;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.BeanFactory;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.data.rest.webmvc.support.DelegatingHandlerMapping;
import org.springframework.web.servlet.HandlerExecutionChain;
import org.springframework.web.servlet.HandlerMapping;


class TSDBPProcess implements BeanPostProcessor {

	private static final Log log = LogFactory.getLog(MethodHandles.lookup().lookupClass());

	private final BeanFactory beanFactory;

	public TSDBPProcess(BeanFactory beanFactory) {
		this.beanFactory = beanFactory;
	}

	@Override
	public Object postProcessBeforeInitialization(Object bean, String beanName)
			throws BeansException {
		if (bean instanceof DelegatingHandlerMapping && !(bean instanceof TraceDelegatingHandlerMapping)) {
			if (log.isDebugEnabled()) {
				log.debug("Wrapping bean [" + beanName + "] of type [" + bean.getClass().getSimpleName() +
						"] in its trace representation");
			}
			return new TraceDelegatingHandlerMapping((DelegatingHandlerMapping) bean,
					this.beanFactory);
		}
		return bean;
	}

	@Override
	public Object postProcessAfterInitialization(Object bean, String beanName)
			throws BeansException {
		return bean;
	}

	private static class TraceDelegatingHandlerMapping extends DelegatingHandlerMapping {

		private final DelegatingHandlerMapping delegate;
		private final BeanFactory beanFactory;

		public TraceDelegatingHandlerMapping(DelegatingHandlerMapping delegate,
				BeanFactory beanFactory) {
			super(Collections.<HandlerMapping>emptyList());
			this.delegate = delegate;
			this.beanFactory = beanFactory;
		}

		@Override
		public int getOrder() {
			return this.delegate.getOrder();
		}

		@Override
		public HandlerExecutionChain getHandler(HttpServletRequest request)
				throws Exception {
			HandlerExecutionChain handlerExecutionChain = this.delegate.getHandler(request);
			if (handlerExecutionChain == null) {
				return null;
			}
			handlerExecutionChain.addInterceptor(new THInter(this.beanFactory));
			return handlerExecutionChain;
		}
	}
}
