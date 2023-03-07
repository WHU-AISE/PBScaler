

package org.myproject.ms.monitoring.antn;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;

import org.aopalliance.intercept.MethodInvocation;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.springframework.aop.support.AopUtils;
import org.springframework.beans.factory.BeanFactory;
import org.myproject.ms.monitoring.Chainer;
import org.springframework.util.StringUtils;


class SpanTagAnnotationHandler {

	private static final Log log = LogFactory.getLog(MethodHandles.lookup().lookupClass());

	private final BeanFactory beanFactory;
	private Chainer tracer;
	
	SpanTagAnnotationHandler(BeanFactory beanFactory) {
		this.beanFactory = beanFactory;
	}

	void addAnnotatedParameters(MethodInvocation pjp) {
		try {
			Method method = pjp.getMethod();
			Method mostSpecificMethod = AopUtils.getMostSpecificMethod(method,
					pjp.getThis().getClass());
			List<SleuthAnnotatedParameter> annotatedParameters =
					SleuthAnnotationUtils.findAnnotatedParameters(mostSpecificMethod, pjp.getArguments());
			getAnnotationsFromInterfaces(pjp, mostSpecificMethod, annotatedParameters);
			mergeAnnotatedMethodsIfNecessary(pjp, method, mostSpecificMethod,
					annotatedParameters);
			addAnnotatedArguments(annotatedParameters);
		} catch (SecurityException e) {
			log.error("Exception occurred while trying to add annotated parameters", e);
		}
	}

	private void getAnnotationsFromInterfaces(MethodInvocation pjp,
			Method mostSpecificMethod,
			List<SleuthAnnotatedParameter> annotatedParameters) {
		Class<?>[] implementedInterfaces = pjp.getThis().getClass().getInterfaces();
		if (implementedInterfaces.length > 0) {
			for (Class<?> implementedInterface : implementedInterfaces) {
				for (Method methodFromInterface : implementedInterface.getMethods()) {
					if (methodsAreTheSame(mostSpecificMethod, methodFromInterface)) {
						List<SleuthAnnotatedParameter> annotatedParametersForActualMethod =
								SleuthAnnotationUtils.findAnnotatedParameters(methodFromInterface, pjp.getArguments());
						mergeAnnotatedParameters(annotatedParameters, annotatedParametersForActualMethod);
					}
				}
			}
		}
	}

	private boolean methodsAreTheSame(Method mostSpecificMethod, Method method1) {
		return method1.getName().equals(mostSpecificMethod.getName()) &&
				Arrays.equals(method1.getParameterTypes(), mostSpecificMethod.getParameterTypes());
	}

	private void mergeAnnotatedMethodsIfNecessary(MethodInvocation pjp, Method method,
			Method mostSpecificMethod, List<SleuthAnnotatedParameter> annotatedParameters) {
		// that can happen if we have an abstraction and a concrete class that is
		// annotated with @NewSpan annotation
		if (!method.equals(mostSpecificMethod)) {
			List<SleuthAnnotatedParameter> annotatedParametersForActualMethod = SleuthAnnotationUtils.findAnnotatedParameters(
					method, pjp.getArguments());
			mergeAnnotatedParameters(annotatedParameters, annotatedParametersForActualMethod);
		}
	}

	private void mergeAnnotatedParameters(List<SleuthAnnotatedParameter> annotatedParametersIndices,
			List<SleuthAnnotatedParameter> annotatedParametersIndicesForActualMethod) {
		for (SleuthAnnotatedParameter container : annotatedParametersIndicesForActualMethod) {
			final int index = container.parameterIndex;
			boolean parameterContained = false;
			for (SleuthAnnotatedParameter parameterContainer : annotatedParametersIndices) {
				if (parameterContainer.parameterIndex == index) {
					parameterContained = true;
					break;
				}
			}
			if (!parameterContained) {
				annotatedParametersIndices.add(container);
			}
		}
	}

	private void addAnnotatedArguments(List<SleuthAnnotatedParameter> toBeAdded) {
		for (SleuthAnnotatedParameter container : toBeAdded) {
			String tagValue = resolveTagValue(container.annotation, container.argument);
			tracer().addTag(container.annotation.value(), tagValue);
		}
	}

	String resolveTagValue(SpanTag annotation, Object argument) {
		if (argument == null) {
			return "";
		}
		if (annotation.resolver() != NoOpTagValueResolver.class) {
			TagValueResolver tagValueResolver = this.beanFactory.getBean(annotation.resolver());
			return tagValueResolver.resolve(argument);
		} else if (StringUtils.hasText(annotation.expression())) {
			return this.beanFactory.getBean(TagValueExpressionResolver.class)
					.resolve(annotation.expression(), argument);
		}
		return argument.toString();
	}

	private Chainer tracer() {
		if (this.tracer == null) {
			this.tracer = this.beanFactory.getBean(Chainer.class);
		}
		return this.tracer;
	}

}
