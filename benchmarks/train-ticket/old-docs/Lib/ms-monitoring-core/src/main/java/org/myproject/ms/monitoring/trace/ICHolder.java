

package org.myproject.ms.monitoring.trace;

import org.apache.commons.logging.Log;
import org.myproject.ms.monitoring.Item;
import org.springframework.core.NamedThreadLocal;


class ICHolder {

	private static final Log log = org.apache.commons.logging.LogFactory
			.getLog(ICHolder.class);
	private static final ThreadLocal<SpanContext> CURRENT_SPAN = new NamedThreadLocal<>(
			"Trace Context");

	
	static Item getCurrentSpan() {
		return isTracing() ? CURRENT_SPAN.get().span : null;
	}

	
	static void setCurrentSpan(Item span) {
		if (log.isTraceEnabled()) {
			log.trace("Setting current span " + span);
		}
		push(span, false);
	}

	
	static void removeCurrentSpan() {
		CURRENT_SPAN.remove();
	}

	
	static boolean isTracing() {
		return CURRENT_SPAN.get() != null;
	}

	
	static void close(SpanFunction spanFunction) {
		SpanContext current = CURRENT_SPAN.get();
		CURRENT_SPAN.remove();
		while (current != null) {
			current = current.parent;
			spanFunction.apply(current != null ? current.span : null);
			if (current != null) {
				if (!current.autoClose) {
					CURRENT_SPAN.set(current);
					current = null;
				}
			}
		}
	}

	
	static void close() {
		close(new NoOpFunction());
	}

	
	static void push(Item span, boolean autoClose) {
		if (isCurrent(span)) {
			return;
		}
		CURRENT_SPAN.set(new SpanContext(span, autoClose));
	}

	private static boolean isCurrent(Item span) {
		if (span == null || CURRENT_SPAN.get() == null) {
			return false;
		}
		return span.equals(CURRENT_SPAN.get().span);
	}

	private static class SpanContext {
		Item span;
		boolean autoClose;
		SpanContext parent;

		public SpanContext(Item span, boolean autoClose) {
			this.span = span;
			this.autoClose = autoClose;
			this.parent = CURRENT_SPAN.get();
		}
	}

	interface SpanFunction {
		void apply(Item span);
	}

	private static class NoOpFunction implements SpanFunction {
		@Override public void apply(Item span) { }
	}
}
