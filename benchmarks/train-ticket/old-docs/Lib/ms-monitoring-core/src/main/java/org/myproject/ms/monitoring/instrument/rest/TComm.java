

package org.myproject.ms.monitoring.instrument.rest;

import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.ChainKeys;

import com.netflix.hystrix.HystrixCommand;


public abstract class TComm<R> extends HystrixCommand<R> {

	private static final String HYSTRIX_COMPONENT = "hystrix";

	private final Chainer tracer;
	private final ChainKeys traceKeys;
	private final Item parentSpan;

	protected TComm(Chainer tracer, ChainKeys traceKeys, Setter setter) {
		super(setter);
		this.tracer = tracer;
		this.traceKeys = traceKeys;
		this.parentSpan = tracer.getCurrentSpan();
	}

	@Override
	protected R run() throws Exception {
		String commandKeyName = getCommandKey().name();
		Item span = startSpan(commandKeyName);
		this.tracer.addTag(Item.SPAN_LOCAL_COMPONENT_TAG_NAME, HYSTRIX_COMPONENT);
		this.tracer.addTag(this.traceKeys.getHystrix().getPrefix() +
				this.traceKeys.getHystrix().getCommandKey(), commandKeyName);
		this.tracer.addTag(this.traceKeys.getHystrix().getPrefix() +
				this.traceKeys.getHystrix().getCommandGroup(), getCommandGroup().name());
		this.tracer.addTag(this.traceKeys.getHystrix().getPrefix() +
				this.traceKeys.getHystrix().getThreadPoolKey(), getThreadPoolKey().name());
		try {
			return doRun();
		}
		finally {
			close(span);
		}
	}

	private Item startSpan(String commandKeyName) {
		Item span = this.parentSpan;
		if (span == null) {
			return this.tracer.createSpan(commandKeyName, this.parentSpan);
		}
		return this.tracer.continueSpan(span);
	}

	private void close(Item span) {
		if (this.parentSpan == null) {
			this.tracer.close(span);
		} else {
			this.tracer.detach(span);
		}
	}

	public abstract R doRun() throws Exception;
}
