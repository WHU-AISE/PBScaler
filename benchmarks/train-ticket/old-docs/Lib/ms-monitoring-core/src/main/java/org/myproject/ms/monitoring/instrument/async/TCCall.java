

package org.myproject.ms.monitoring.instrument.async;

import java.util.concurrent.Callable;

import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.ItemNamer;
import org.myproject.ms.monitoring.ChainCallable;
import org.myproject.ms.monitoring.Chainer;


@Deprecated
public class TCCall<V> extends ChainCallable<V> implements Callable<V> {

	public TCCall(Chainer tracer, ItemNamer spanNamer, Callable<V> delegate) {
		super(tracer, spanNamer, delegate);
	}

	@Override
	protected Item startSpan() {
		return getTracer().continueSpan(getParent());
	}

	@Override
	protected void close(Item span) {
		if (getTracer().isTracing()) {
			getTracer().detach(span);
		}
	}
}
