

package org.myproject.ms.monitoring.antn;

import java.lang.annotation.Annotation;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.springframework.core.annotation.AnnotationUtils;


class SleuthAnnotationUtils {

	private static final Log log = LogFactory.getLog(MethodHandles.lookup().lookupClass());

	static boolean isMethodAnnotated(Method method) {
		return findAnnotation(method, NewSpan.class) != null ||
				findAnnotation(method, ContinueSpan.class) != null;
	}

	static boolean hasAnnotatedParams(Method method, Object[] args) {
		return !findAnnotatedParameters(method, args).isEmpty();
	}

	static List<SleuthAnnotatedParameter> findAnnotatedParameters(Method method, Object[] args) {
		Annotation[][] parameters = method.getParameterAnnotations();
		List<SleuthAnnotatedParameter> result = new ArrayList<>();
		int i = 0;
		for (Annotation[] parameter : parameters) {
			for (Annotation parameter2 : parameter) {
				if (parameter2 instanceof SpanTag) {
					result.add(new SleuthAnnotatedParameter(i, (SpanTag) parameter2, args[i]));
				}
			}
			i++;
		}
		return result;
	}

	
	static <T extends Annotation> T findAnnotation(Method method, Class<T> clazz) {
		T annotation = AnnotationUtils.findAnnotation(method, clazz);
		if (annotation == null) {
			try {
				annotation = AnnotationUtils.findAnnotation(
						method.getDeclaringClass().getMethod(method.getName(),
								method.getParameterTypes()), clazz);
			} catch (NoSuchMethodException | SecurityException e) {
				if (log.isDebugEnabled()) {
					log.debug("Exception occurred while tyring to find the annotation", e);
				}
			}
		}
		return annotation;
	}
}
