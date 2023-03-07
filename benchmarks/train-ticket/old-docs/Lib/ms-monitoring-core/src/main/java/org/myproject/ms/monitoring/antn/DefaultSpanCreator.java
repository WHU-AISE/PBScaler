
package org.myproject.ms.monitoring.antn;

import java.lang.invoke.MethodHandles;

import org.aopalliance.intercept.MethodInvocation;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.util.ItemNameUtil;
import org.springframework.util.StringUtils;


class DefaultSpanCreator implements SpanCreator {

	private static final Log log = LogFactory.getLog(MethodHandles.lookup().lookupClass());

	private final Chainer tracer;

	DefaultSpanCreator(Chainer tracer) {
		this.tracer = tracer;
	}

	@Override public Item createSpan(MethodInvocation pjp, NewSpan newSpanAnnotation) {
		String name = StringUtils.isEmpty(newSpanAnnotation.name()) ?
				pjp.getMethod().getName() : newSpanAnnotation.name();
		String changedName = ItemNameUtil.toLowerHyphen(name);
		if (log.isDebugEnabled()) {
			log.debug("For the class [" + pjp.getThis().getClass() + "] method "
					+ "[" + pjp.getMethod().getName() + "] will name the span [" + changedName + "]");
		}
		return createSpan(changedName);
	}

	private Item createSpan(String name) {
		if (this.tracer.isTracing()) {
			return this.tracer.createSpan(name, this.tracer.getCurrentSpan());
		}
		return this.tracer.createSpan(name);
	}

}
