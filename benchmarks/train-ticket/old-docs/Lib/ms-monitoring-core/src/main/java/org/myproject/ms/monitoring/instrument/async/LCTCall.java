

package org.myproject.ms.monitoring.instrument.async;

import java.util.concurrent.Callable;

import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.ItemNamer;
import org.myproject.ms.monitoring.ChainCallable;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.ChainKeys;


public class LCTCall<V> extends ChainCallable<V> {

	protected static final String ASYNC_COMPONENT = "async";

	private final ChainKeys traceKeys;

	public LCTCall(Chainer tracer, ChainKeys traceKeys,
			ItemNamer spanNamer, Callable<V> delegate) {
		super(tracer, spanNamer, delegate);
		this.traceKeys = traceKeys;
	}

	public LCTCall(Chainer tracer, ChainKeys traceKeys,
			ItemNamer spanNamer, String name, Callable<V> delegate) {
		super(tracer, spanNamer, delegate, name);
		this.traceKeys = traceKeys;
	}

	@Override
	public V call() throws Exception {
		Item span = startSpan();
		try {
			return this.getDelegate().call();
		}
		finally {
			close(span);
		}
	}

	@Override
	protected Item startSpan() {
		Item span = getTracer().createSpan(getSpanName(), getParent());
		getTracer().addTag(Item.SPAN_LOCAL_COMPONENT_TAG_NAME, ASYNC_COMPONENT);
		getTracer().addTag(this.traceKeys.getAsync().getPrefix() +
				this.traceKeys.getAsync().getThreadNameKey(), Thread.currentThread().getName());
		return span;
	}
}
