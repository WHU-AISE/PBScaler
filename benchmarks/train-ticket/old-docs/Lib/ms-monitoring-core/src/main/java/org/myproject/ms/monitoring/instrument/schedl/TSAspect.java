

package org.myproject.ms.monitoring.instrument.schedl;

import java.util.regex.Pattern;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.ChainKeys;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.util.ItemNameUtil;


@Aspect
public class TSAspect {

	private static final String SCHEDULED_COMPONENT = "scheduled";

	private final Chainer tracer;
	private final ChainKeys traceKeys;
	private final Pattern skipPattern;

	public TSAspect(Chainer tracer, ChainKeys traceKeys, Pattern skipPattern) {
		this.tracer = tracer;
		this.traceKeys = traceKeys;
		this.skipPattern = skipPattern;
	}

	@Around("execution (@org.springframework.scheduling.annotation.Scheduled  * *.*(..))")
	public Object traceBackgroundThread(final ProceedingJoinPoint pjp) throws Throwable {
		if (this.skipPattern.matcher(pjp.getTarget().getClass().getName()).matches()) {
			return pjp.proceed();
		}
		String spanName = ItemNameUtil.toLowerHyphen(pjp.getSignature().getName());
		Item span = this.tracer.createSpan(spanName);
		this.tracer.addTag(Item.SPAN_LOCAL_COMPONENT_TAG_NAME, SCHEDULED_COMPONENT);
		this.tracer.addTag(this.traceKeys.getAsync().getPrefix() +
				this.traceKeys.getAsync().getClassNameKey(), pjp.getTarget().getClass().getSimpleName());
		this.tracer.addTag(this.traceKeys.getAsync().getPrefix() +
				this.traceKeys.getAsync().getMethodNameKey(), pjp.getSignature().getName());
		try {
			return pjp.proceed();
		}
		finally {
			this.tracer.close(span);
		}
	}

}
