

package org.myproject.ms.monitoring.instrument.async;

import java.util.concurrent.Executor;

import org.springframework.aop.interceptor.AsyncUncaughtExceptionHandler;
import org.springframework.beans.factory.BeanFactory;
import org.springframework.scheduling.annotation.AsyncConfigurer;
import org.springframework.scheduling.annotation.AsyncConfigurerSupport;


public class LTACus extends AsyncConfigurerSupport {

	private final BeanFactory beanFactory;
	private final AsyncConfigurer delegate;

	public LTACus(BeanFactory beanFactory, AsyncConfigurer delegate) {
		this.beanFactory = beanFactory;
		this.delegate = delegate;
	}

	@Override
	public Executor getAsyncExecutor() {
		return new LTExec(this.beanFactory, this.delegate.getAsyncExecutor());
	}

	@Override
	public AsyncUncaughtExceptionHandler getAsyncUncaughtExceptionHandler() {
		return this.delegate.getAsyncUncaughtExceptionHandler();
	}

}
