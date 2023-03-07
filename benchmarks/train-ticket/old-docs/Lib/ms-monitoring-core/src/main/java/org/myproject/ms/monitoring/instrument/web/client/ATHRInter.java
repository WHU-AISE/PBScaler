

package org.myproject.ms.monitoring.instrument.web.client;

import java.lang.invoke.MethodHandles;
import java.net.URI;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.myproject.ms.monitoring.instrument.web.HSInject;
import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.Chainer;
import org.myproject.ms.monitoring.instrument.web.HTKInject;
import org.myproject.ms.monitoring.util.ItemNameUtil;
import org.springframework.http.HttpRequest;

abstract class ATHRInter {

	protected static final Log log = LogFactory.getLog(MethodHandles.lookup().lookupClass());

	protected final Chainer tracer;
	protected final HSInject spanInjector;
	protected final HTKInject keysInjector;

	protected ATHRInter(Chainer tracer,
			HSInject spanInjector, HTKInject keysInjector) {
		this.tracer = tracer;
		this.spanInjector = spanInjector;
		this.keysInjector = keysInjector;
	}

	
	protected void publishStartEvent(HttpRequest request) {
		URI uri = request.getURI();
		String spanName = getName(uri);
		Item newSpan = this.tracer.createSpan(spanName);
		this.spanInjector.inject(newSpan, new HRTMap(request));
		addRequestTags(request);
		newSpan.logEvent(Item.CLIENT_SEND);
		if (log.isDebugEnabled()) {
			log.debug("Starting new client span [" + newSpan + "]");
		}
	}

	private String getName(URI uri) {
		return ItemNameUtil.shorten(uriScheme(uri) + ":" + uri.getPath());
	}

	private String uriScheme(URI uri) {
		return uri.getScheme() == null ? "http" : uri.getScheme();
	}

	
	protected void addRequestTags(HttpRequest request) {
		this.keysInjector.addRequestTags(request.getURI().toString(),
				request.getURI().getHost(),
				request.getURI().getPath(),
				request.getMethod().name(),
				request.getHeaders());
	}

	
	public void finish() {
		if (!isTracing()) {
			return;
		}
		currentSpan().logEvent(Item.CLIENT_RECV);
		this.tracer.close(this.currentSpan());
	}

	protected Item currentSpan() {
		return this.tracer.getCurrentSpan();
	}

	protected boolean isTracing() {
		return this.tracer.isTracing();
	}

}
