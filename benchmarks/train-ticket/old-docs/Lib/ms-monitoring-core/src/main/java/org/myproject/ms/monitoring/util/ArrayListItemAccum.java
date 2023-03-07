

package org.myproject.ms.monitoring.util;

import java.util.ArrayList;
import java.util.List;

import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.ItemReporter;


public class ArrayListItemAccum implements ItemReporter {
	private final List<Item> spans = new ArrayList<>();

	public List<Item> getSpans() {
		synchronized (this.spans) {
			return this.spans;
		}
	}

	@Override
	public String toString() {
		return "ArrayListSpanAccumulator{" +
				"spans=" + getSpans() +
				'}';
	}

	@Override
	public void report(Item span) {
		synchronized (this.spans) {
			this.spans.add(span);
		}
	}

	public void clear() {
		synchronized (this.spans) {
			this.spans.clear();
		}
	}
}
