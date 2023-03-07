

package org.myproject.ms.monitoring.instrument.async;

import java.lang.reflect.Method;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.beans.factory.BeanFactory;
import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.ChainKeys;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.util.ItemNameUtil;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.util.ReflectionUtils;


@Aspect
public class TAAsp {

	private static final String ASYNC_COMPONENT = "async";

	private final Chainer tracer;
	private final ChainKeys traceKeys;
	private final BeanFactory beanFactory;

	public TAAsp(Chainer tracer, ChainKeys traceKeys, BeanFactory beanFactory) {
		this.tracer = tracer;
		this.traceKeys = traceKeys;
		this.beanFactory = beanFactory;
	}

	@Around("execution (@org.springframework.scheduling.annotation.Async  * *.*(..))")
	public Object traceBackgroundThread(final ProceedingJoinPoint pjp) throws Throwable {
		Item span = this.tracer.createSpan(
				ItemNameUtil.toLowerHyphen(pjp.getSignature().getName()));
		this.tracer.addTag(Item.SPAN_LOCAL_COMPONENT_TAG_NAME, ASYNC_COMPONENT);
		this.tracer.addTag(this.traceKeys.getAsync().getPrefix() +
				this.traceKeys.getAsync().getClassNameKey(), pjp.getTarget().getClass().getSimpleName());
		this.tracer.addTag(this.traceKeys.getAsync().getPrefix() +
				this.traceKeys.getAsync().getMethodNameKey(), pjp.getSignature().getName());
		try {
			return pjp.proceed();
		} finally {
			this.tracer.close(span);
		}
	}

	@Around("execution (* org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor.*(..))")
	public Object traceThreadPoolTaskExecutor(final ProceedingJoinPoint pjp) throws Throwable {
		LTTPTExec executor = new LTTPTExec(this.beanFactory,
				(ThreadPoolTaskExecutor) pjp.getTarget());
		Method methodOnTracedBean = getMethod(pjp, executor);
		if (methodOnTracedBean != null) {
			return methodOnTracedBean.invoke(executor, pjp.getArgs());
		}
		return pjp.proceed();
	}

	private Method getMethod(ProceedingJoinPoint pjp, Object object) {
		MethodSignature signature = (MethodSignature) pjp.getSignature();
		Method method = signature.getMethod();
		return ReflectionUtils
				.findMethod(object.getClass(), method.getName(), method.getParameterTypes());
	}

}
