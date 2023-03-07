

package org.myproject.ms.monitoring.instrument.async;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.concurrent.Executor;

import org.aopalliance.intercept.MethodInterceptor;
import org.aopalliance.intercept.MethodInvocation;
import org.springframework.aop.framework.ProxyFactoryBean;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.BeanFactory;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.util.ReflectionUtils;


class EBPProc implements BeanPostProcessor {

	private final BeanFactory beanFactory;

	EBPProc(BeanFactory beanFactory) {
		this.beanFactory = beanFactory;
	}

	@Override
	public Object postProcessBeforeInitialization(Object bean, String beanName)
			throws BeansException {
		return bean;
	}

	@Override
	public Object postProcessAfterInitialization(Object bean, String beanName)
			throws BeansException {
		if (bean instanceof Executor && !(bean instanceof ThreadPoolTaskExecutor)) {
			Method execute = ReflectionUtils.findMethod(bean.getClass(), "execute", Runnable.class);
			boolean methodFinal = Modifier.isFinal(execute.getModifiers());
			boolean classFinal = Modifier.isFinal(bean.getClass().getModifiers());
			boolean cglibProxy = !methodFinal && !classFinal;
			Executor executor = (Executor) bean;
			ProxyFactoryBean factory = new ProxyFactoryBean();
			factory.setProxyTargetClass(cglibProxy);
			factory.addAdvice(new ExecutorMethodInterceptor(executor, this.beanFactory));
			factory.setTarget(bean);
			return factory.getObject();
		}
		return bean;
	}
}

class ExecutorMethodInterceptor implements MethodInterceptor {

	private final Executor delegate;
	private final BeanFactory beanFactory;

	ExecutorMethodInterceptor(Executor delegate, BeanFactory beanFactory) {
		this.delegate = delegate;
		this.beanFactory = beanFactory;
	}

	@Override public Object invoke(MethodInvocation invocation)
			throws Throwable {
		LTExec executor = new LTExec(this.beanFactory, this.delegate);
		Method methodOnTracedBean = getMethod(invocation, executor);
		if (methodOnTracedBean != null) {
			return methodOnTracedBean.invoke(executor, invocation.getArguments());
		}
		return invocation.proceed();
	}

	private Method getMethod(MethodInvocation invocation, Object object) {
		Method method = invocation.getMethod();
		return ReflectionUtils
				.findMethod(object.getClass(), method.getName(), method.getParameterTypes());
	}
}
